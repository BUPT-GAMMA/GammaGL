"""Native TensorLayerX GTE encoder for serialized graph tokens."""

from __future__ import annotations

import tensorlayerx as tlx

from ._graph_transformer import (
    _parameter_count,
    masked_mean,
    merge_heads,
    scaled_dot_product_attention,
    split_heads,
)


def _linear(in_features, out_features, bias=True):
    # NewPreTrainedModel._init_weights uses an untruncated normal distribution.
    return tlx.nn.Linear(in_features=in_features, out_features=out_features,
                         W_init=tlx.initializers.RandomNormal(stddev=0.02),
                         b_init=tlx.initializers.Constant(0.0) if bias else None)


def rotate_half(value):
    width = int(tlx.get_tensor_shape(value)[-1])
    if width % 2 != 0:
        raise ValueError("RoPE head dimension must be even.")
    half = width // 2
    first = value[..., :half]
    second = value[..., half:]
    return tlx.concat([-second, first], axis=-1)


def apply_rotary_pos_emb(query, key, cosine, sine):
    return (
        query * cosine + rotate_half(query) * sine,
        key * cosine + rotate_half(key) * sine,
    )


class RotaryEmbedding(tlx.nn.Module):
    """Official-GTE-compatible NTK-scaled rotary position embedding."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 20000.0,
        scaling_factor: float = 8.0,
        name: str | None = None,
    ):
        super().__init__(name=name)
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("RoPE dimension must be a positive even integer.")
        if scaling_factor <= 0:
            raise ValueError("RoPE scaling_factor must be positive.")
        self.dim = int(dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.base = float(base)
        self.scaling_factor = float(scaling_factor)

    def forward(self, reference):
        sequence_length = int(tlx.get_tensor_shape(reference)[1])
        if sequence_length > self.max_position_embeddings:
            raise ValueError("Input sequence length exceeds max_position_embeddings.")
        half_reference = reference[0, 0, : self.dim // 2]
        dimension_steps = (
            tlx.cumsum(tlx.ones_like(half_reference), axis=0) - 1.0) * 2.0
        exponent = dimension_steps / float(self.dim)
        adjusted_base = self.base * self.scaling_factor
        base_tensor = tlx.ones_like(dimension_steps) * adjusted_base
        inverse_frequency = 1.0 / tlx.pow(base_tensor, exponent)
        inverse_frequency = inverse_frequency / (
            self.scaling_factor ** (2.0 / float(self.dim)))
        position_reference = reference[0, :, 0]
        positions = tlx.cumsum(tlx.ones_like(position_reference), axis=0) - 1.0
        frequencies = tlx.einsum("i,j->ij", positions, inverse_frequency)
        angles = tlx.concat([frequencies, frequencies], axis=-1)
        cosine = tlx.expand_dims(tlx.expand_dims(tlx.cos(angles), axis=0), axis=0)
        sine = tlx.expand_dims(tlx.expand_dims(tlx.sin(angles), axis=0), axis=0)
        return cosine, sine


class GraphGTESelfAttention(tlx.nn.Module):
    """Packed-QKV self-attention with RoPE applied to query and key."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout_rate: float = 0.1,
        name: str | None = None,
    ):
        super().__init__(name=name)
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.head_size = self.hidden_size // self.num_attention_heads
        if self.head_size % 2 != 0:
            raise ValueError("GTE attention head size must be even for RoPE.")
        self.qkv_proj = _linear(self.hidden_size, self.hidden_size * 3)
        self.out_proj = _linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = tlx.nn.Dropout(p=float(attention_dropout_rate))

    def forward(self, hidden_states, attention_mask=None, rope_embeddings=None):
        if rope_embeddings is None:
            raise ValueError("GTE attention requires rotary embeddings.")
        query, key, value = tlx.split(
            self.qkv_proj(hidden_states), 3, axis=-1)
        query = split_heads(query, self.num_attention_heads, self.head_size)
        key = split_heads(key, self.num_attention_heads, self.head_size)
        value = split_heads(value, self.num_attention_heads, self.head_size)
        query, key = apply_rotary_pos_emb(
            query, key, rope_embeddings[0], rope_embeddings[1])
        context = scaled_dot_product_attention(
            query, key, value, attention_mask, dropout=self.attention_dropout)
        return self.out_proj(merge_heads(context, self.hidden_size))


class GraphGatedMLP(tlx.nn.Module):
    """GTE gated GELU feed-forward network."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout_rate: float = 0.1,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.intermediate_size = int(intermediate_size)
        self.up_gate_proj = _linear(
            hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = _linear(self.intermediate_size, hidden_size)
        self.hidden_dropout = tlx.nn.Dropout(p=float(dropout_rate))

    def forward(self, hidden_states):
        up_states, gate = tlx.split(
            self.up_gate_proj(hidden_states), 2, axis=-1)
        gated_states = up_states * tlx.gelu(gate)
        return self.down_proj(self.hidden_dropout(gated_states))


class GraphGTELayer(tlx.nn.Module):
    """Post-normalization GTE attention and gated-MLP block."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float | None = None,
        layer_norm_eps: float = 1e-12,
        name: str | None = None,
    ):
        super().__init__(name=name)
        attention_dropout_rate = (
            dropout_rate if attention_dropout_rate is None
            else attention_dropout_rate
        )
        self.attention = GraphGTESelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout_rate=attention_dropout_rate,
        )
        self.mlp = GraphGatedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
        )
        self.attention_norm = tlx.nn.LayerNorm(
            normalized_shape=hidden_size,
            epsilon=layer_norm_eps,
            gamma_init="ones",
            beta_init="zeros",
        )
        self.mlp_norm = tlx.nn.LayerNorm(
            normalized_shape=hidden_size,
            epsilon=layer_norm_eps,
            gamma_init="ones",
            beta_init="zeros",
        )
        self.hidden_dropout = tlx.nn.Dropout(p=float(dropout_rate))

    def forward(self, hidden_states, attention_mask=None, rope_embeddings=None):
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            rope_embeddings=rope_embeddings,
        )
        hidden_states = self.attention_norm(
            hidden_states + self.hidden_dropout(attention_output))
        mlp_output = self.mlp(hidden_states)
        return self.mlp_norm(hidden_states + self.hidden_dropout(mlp_output))


class GraphGTE(tlx.nn.Module):
    r"""GTE graph token encoder with packed QKV, rotary positions and gated GELU.

    Used in `"Graph Tokenization for Bridging Graphs and Transformers"
    <https://arxiv.org/abs/2603.11099>`_. The native TLX implementation follows
    the `pinned official GTE source
    <https://huggingface.co/Alibaba-NLP/gte-multilingual-base/tree/9bbca17d9273fd0d03d5725c7a4b0f6b45142062>`_.
    Supports the TLX torch and TensorFlow eager training paths. Construction
    is random initialization; use :meth:`from_pretrained` for official
    encoder weights, including resized and vocabulary-aligned word embeddings.
    MLM and graph heads remain new. :meth:`from_official_config` constructs
    the pinned architecture randomly, as in the AQSOL/GTE preset.

    Parameters
    ----------
    vocab_size: int
        Graph model vocabulary size, including special tokens.
    output_dim: int
        Number of regression targets or classes.
    hidden_size: int
        Encoder width. (default: ``768``)
    num_hidden_layers: int
        Number of encoder blocks. (default: ``12``)
    num_attention_heads: int
        Must divide hidden_size and give an even head width for RoPE.
        (default: ``12``)
    intermediate_size: int
        Gated feed-forward width. (default: ``3072``)
    max_position_embeddings: int
        Maximum sequence length including special tokens.
        (default: ``8192``)
    dropout_rate: float
        Embedding and hidden dropout. (default: ``0.1``)
    attention_dropout_rate: float
        None uses dropout_rate; from_pretrained uses the pinned config's
        attention dropout (0.0). (default: :obj:`None`)
    layer_norm_eps: float
        Layer normalization epsilon. (default: ``1e-12``)
    pad_token_id: int
        Zero embedding row with no lookup gradient. attention_mask separately
        controls attention / pooling visibility. The pinned pretrained config
        uses 1. (default: ``0``)
    type_vocab_size: int
        Token-type embedding vocabulary; inputs use type 0.
        (default: ``1``)
    task_type: str
        Task metadata; outputs remain raw scores. (default: ``"regression"``)
    pooling: str
        "mean" or "cls", as in :class:`GraphTokenTransformer`.
        (default: ``"mean"``)
    task_dropout: float
        Graph task head dropout. (default: ``0.1``)
    rope_theta: float
        Rotary frequency base. (default: ``20000.0``)
    rope_scaling_factor: float
        Positive official NTK scaling factor. (default: ``8.0``)
    name: str
        TensorLayerX module name. (default: :obj:`None`)

    Raises
    ------
    ValueError
        If pooling, attention dimensions or rotary scaling are
        invalid. Oversized sequences also raise ValueError on forward.

    Examples
    --------
    A small random model for an offline shape check (not a paper run):

    >>> import tensorlayerx as tlx
    >>> from gammagl.models import GraphGTE
    >>> model = GraphGTE(32, 1, hidden_size=16, num_hidden_layers=1,
    ...     num_attention_heads=4, intermediate_size=32)
    >>> model.set_eval()
    >>> ids = tlx.convert_to_tensor([[3, 8, 4]], dtype=tlx.int64)
    >>> tuple(model(ids, tlx.ones_like(ids), task="supervised").shape)
    (1, 1)
    """

    encoder_type = "gte"
    model_name = "gte-base"
    position_embedding_type = "rope"
    hidden_act = "gelu"

    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 8192,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float | None = None,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        type_vocab_size: int = 1,
        task_type: str = "regression",
        pooling: str = "mean",
        task_dropout: float = 0.1,
        rope_theta: float = 20000.0,
        rope_scaling_factor: float = 8.0,
        name: str | None = None,
    ):
        super().__init__(name=name)
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'.")
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        self.vocab_size = int(vocab_size)
        self.output_dim = int(output_dim)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size)
        self.max_position_embeddings = int(max_position_embeddings)
        self.dropout_rate = float(dropout_rate)
        self.attention_dropout_rate = float(
            dropout_rate if attention_dropout_rate is None
            else attention_dropout_rate)
        self.layer_norm_eps = float(layer_norm_eps)
        self.pad_token_id = int(pad_token_id)
        self.type_vocab_size = int(type_vocab_size)
        self.task_type = str(task_type)
        self.pooling = pooling
        self.task_dropout_rate = float(task_dropout)
        self.rope_theta = float(rope_theta)
        self.rope_scaling_factor = float(rope_scaling_factor)
        self.head_size = self.hidden_size // self.num_attention_heads

        self.token_embeddings = tlx.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            E_init=tlx.initializers.RandomNormal(stddev=0.02),
        )
        self.token_type_embeddings = tlx.nn.Embedding(
            num_embeddings=self.type_vocab_size,
            embedding_dim=self.hidden_size,
            E_init=tlx.initializers.RandomNormal(stddev=0.02),
        )
        self.embedding_norm = tlx.nn.LayerNorm(
            normalized_shape=self.hidden_size,
            epsilon=self.layer_norm_eps,
            gamma_init="ones",
            beta_init="zeros",
        )
        self.embedding_dropout = tlx.nn.Dropout(p=self.dropout_rate)
        self.rotary_embedding = RotaryEmbedding(
            dim=self.head_size,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            scaling_factor=self.rope_scaling_factor,
        )
        self.encoder_layers = tlx.nn.ModuleList([
            GraphGTELayer(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                layer_norm_eps=self.layer_norm_eps,
            )
            for _ in range(self.num_hidden_layers)
        ])
        self.mlm_head = _linear(self.hidden_size, self.vocab_size)
        task_hidden_size = max(1, self.hidden_size // 2)
        self.task_head_in = _linear(self.hidden_size, task_hidden_size)
        self.task_head_dropout = tlx.nn.Dropout(p=self.task_dropout_rate)
        self.task_head_out = _linear(task_hidden_size, self.output_dim)
        self.pretrained_manifest = None
        self.initialization_manifest = None
        self.model_id_to_official_id = list(range(self.vocab_size))
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be inside the model vocabulary.")
        from .graph_gte_pretrained import _assign_tensor
        embeddings = tlx.convert_to_numpy(self.token_embeddings.embeddings).copy()
        embeddings[self.pad_token_id] = 0
        _assign_tensor(self.token_embeddings.embeddings, embeddings, "zero PAD embedding")

    @classmethod
    def _from_official_configuration(cls, vocab_size, output_dim, cache_dir, model_id_to_official_id, kwargs):
        from .graph_gte_pretrained import load_official_gte_config

        config = load_official_gte_config(cache_dir=cache_dir)
        mapping = list(range(vocab_size)) if model_id_to_official_id is None else list(model_id_to_official_id)
        if (len(mapping) != vocab_size or any(type(value) is not int for value in mapping)
                or sorted(mapping) != list(range(vocab_size))):
            raise ValueError("GTE vocabulary mapping must be a complete integer bijection.")
        pad_token_id = kwargs.pop("pad_token_id", config["pad_token_id"])
        if not 0 <= pad_token_id < vocab_size:
            raise ValueError("pad_token_id must be inside the model vocabulary.")
        architecture = {
            "hidden_size": config["hidden_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            "intermediate_size": config["intermediate_size"],
            "max_position_embeddings": config["max_position_embeddings"],
            "dropout_rate": config["hidden_dropout_prob"],
            "attention_dropout_rate": config["attention_probs_dropout_prob"],
            "layer_norm_eps": config["layer_norm_eps"],
            "pad_token_id": mapping[pad_token_id],
            "type_vocab_size": config["type_vocab_size"],
            "rope_theta": config["rope_theta"],
            "rope_scaling_factor": config["rope_scaling"]["factor"],
        }
        for key, value in kwargs.items():
            if key in architecture and architecture[key] != value:
                raise RuntimeError(
                    f"Official GTE config mismatch for {key}: "
                    f"expected {architecture[key]!r}, got {value!r}.")
        architecture.update(kwargs)
        return cls(vocab_size=vocab_size, output_dim=output_dim, **architecture), mapping

    @classmethod
    def from_official_config(cls, vocab_size: int, output_dim: int, cache_dir=None,
                             model_id_to_official_id=None, **kwargs):
        r"""Construct the pinned GTE architecture with random normal weights.

        Only config.json is retrieved; no checkpoint is loaded. This matches
        encoder.reset_weights for AQSOL/GTE. Embeddings / linear kernels use
        N(0, 0.02), linear
        biases are zero, normalization scales are one and PAD is zero with
        no lookup gradient. Vocabulary parameters are permuted together.

        Parameters
        ----------
        vocab_size: int
            Graph model vocabulary size including special IDs.
        output_dim: int
            Number of graph targets or classes.
        cache_dir: str or ``path-like``
            Hugging Face configuration cache. (default: :obj:`None`)
        model_id_to_official_id: list[int]
            Complete vocabulary permutation, as in :meth:`from_pretrained`.
            None uses identity. (default: :obj:`None`)
        **kwargs
            Constructor options; architecture overrides must match the pinned
            configuration. pad_token_id uses model IDs.

        Returns
        -------
        GraphGTE
            Random model with initialization_manifest.mode=official_random.
            This metadata does not assert a reproduced benchmark. The forward
            input / output shapes are those documented on the class.

        Raises
        ------
        RuntimeError
            If the pinned configuration or architecture overrides differ.
        ValueError
            If vocabulary mapping or padding ID is invalid.
        ImportError
            If the optional huggingface-hub package is unavailable.

        Examples
        --------
        .. code-block:: python

            # Fetches the pinned config if it is not already cached.
            model = GraphGTE.from_official_config(32, 1, pad_token_id=0)
        """
        from .graph_gte_pretrained import GTE_MODEL_ID, GTE_REVISION, remap_vocabulary

        model, mapping = cls._from_official_configuration(
            vocab_size, output_dim, cache_dir, model_id_to_official_id, kwargs)
        remap_vocabulary(model, mapping)
        model.initialization_manifest = {
            "model_id": GTE_MODEL_ID, "revision": GTE_REVISION,
            "mode": "official_random", "pretrained": False, "reproduction": False,
            "model_id_to_official_id": mapping,
            "graph_embedding_init": "normal(stddev=0.02); zero PAD",
            "task_head_init": "normal(stddev=0.02); zero bias",
        }
        return model

    @classmethod
    def from_pretrained(cls, vocab_size: int, output_dim: int, cache_dir=None,
                        model_id_to_official_id=None, **kwargs):
        r"""Load the pinned official encoder into a native TLX GraphGTE.

        Parameters
        ----------
        vocab_size: int
            Size of the resized graph embedding / MLM vocabulary.
        output_dim: int
            Width of the newly initialized graph task head.
        cache_dir: str or ``path-like``
            Hugging Face download cache. Missing config and safetensors files
            are downloaded. (default: :obj:`None`)
        model_id_to_official_id: list[int]
            Complete model-ID to official vocabulary-ID permutation. Applied
            to resized embeddings and MLM weight / bias rows together. None
            uses identity. pad_token_id uses model IDs. (default: :obj:`None`)
        **kwargs
            Constructor options. Architecture fields must exactly match the
            pinned official configuration; task_type, pooling, task_dropout and
            name may configure the graph task.

        Returns
        -------
        GraphGTE
            Native model with pretrained_manifest recording revision, SHA-256
            and complete encoder tensor coverage, including resized embeddings.
            Rows in the source vocabulary are retained; added rows and the MLM
            / graph task heads use normal initialization. PAD is zero with no
            lookup gradient. Initialization alone does not establish reproduction.

        Raises
        ------
        RuntimeError
            If the config / requested architecture or checkpoint hash differs,
            or required encoder tensors are missing, unexpected or incompatible.
        ValueError
            If vocabulary mapping or padding ID is invalid.
        ImportError
            If optional huggingface-hub or safetensors dependencies are absent.

        Notes
        -----
        Uses Alibaba-NLP/gte-multilingual-base revision
        ``9bbca17d9273fd0d03d5725c7a4b0f6b45142062``. This initializes the encoder;
        it is not a compatible replacement for a fine-tuned graph checkpoint.
        """
        from .graph_gte_pretrained import (
            GTE_CHECKPOINT, GTE_MODEL_ID, GTE_REVISION,
            download_official_gte_checkpoint, load_pretrained_encoder, remap_vocabulary,
        )
        model, mapping = cls._from_official_configuration(
            vocab_size, output_dim, cache_dir, model_id_to_official_id, kwargs)
        path, actual_hash = download_official_gte_checkpoint(cache_dir=cache_dir)
        report = load_pretrained_encoder(model, path)
        remap_vocabulary(model, mapping)
        model.pretrained_manifest = {
            "model_id": GTE_MODEL_ID, "revision": GTE_REVISION,
            "checkpoint_files": [GTE_CHECKPOINT], "checkpoint_sha256": {GTE_CHECKPOINT: actual_hash},
            "pretrained": True, "mode": "official_pretrained",
            "model_id_to_official_id": mapping,
            "encoder_expected_tensors": len(report["expected"]),
            "encoder_loaded_tensors": len(report["loaded"]),
            "encoder_coverage": report["coverage"],
            "missing": report["missing"], "unexpected": report["unexpected"],
            "ignored": report["ignored"],
            "ignored_checkpoint_keys": report["ignored"],
            "shape_mismatches": report["shape_mismatches"],
            "graph_embedding_init": "official_resize_and_remap",
            "task_head_init": "normal(stddev=0.02); zero bias",
            "reproduction": False,
        }
        model.initialization_manifest = model.pretrained_manifest
        return model

    def _embed(self, input_ids):
        sequence_length = int(tlx.get_tensor_shape(input_ids)[1])
        if sequence_length > self.max_position_embeddings:
            raise ValueError("Input sequence length exceeds max_position_embeddings.")
        word_embeddings = self.token_embeddings(input_ids)
        # Match torch.nn.Embedding(padding_idx=...): PAD stays zero and receives
        # no lookup gradient, including an ordinary raw-0 token aliased to PAD.
        visible = tlx.cast(tlx.expand_dims(input_ids != self.pad_token_id, -1), word_embeddings.dtype)
        word_embeddings = word_embeddings * visible
        token_type_embeddings = self.token_type_embeddings(input_ids * 0)
        return self.embedding_dropout(
            self.embedding_norm(word_embeddings + token_type_embeddings))

    def _embed_inputs(self, inputs_embeds):
        """Apply the official GTE embedding post-processing to raw embeddings.

        This intentionally accepts raw embeddings rather than graph token IDs,
        so the official Hugging Face encoder can be used only as a test oracle.
        """
        sequence_length = int(tlx.get_tensor_shape(inputs_embeds)[1])
        if sequence_length > self.max_position_embeddings:
            raise ValueError("Input sequence length exceeds max_position_embeddings.")
        token_type_ids = tlx.zeros(
            [int(tlx.get_tensor_shape(inputs_embeds)[0]), sequence_length],
            dtype=tlx.int64,
        )
        return self.embedding_dropout(self.embedding_norm(
            inputs_embeds + self.token_type_embeddings(token_type_ids)))

    def encode_embeddings(self, inputs_embeds, attention_mask=None,
                          return_hidden_states=False):
        r"""Run the encoder from raw embeddings, primarily for reference comparisons.

        Parameters
        ----------
        inputs_embeds: ``Tensor``
            Raw floating embeddings of shape (B, L, hidden_size). Token-type
            embeddings and embedding normalization / dropout are still applied.
        attention_mask: ``Tensor``
            Visible-token mask (B, L). (default: :obj:`None`)
        return_hidden_states: bool
            Also return every layer's states. (default: :obj:`False`)

        Returns
        -------
        ``Tensor`` or tuple
            Last hidden state (B, L, hidden_size), or a pair of that tensor and
            a tuple of num_hidden_layers + 1 tensors, including embedding output.

        Raises
        ------
        ValueError
            If L exceeds the configured position capacity.
        """
        hidden_states = self._embed_inputs(inputs_embeds)
        hidden_states_list = [hidden_states]
        rope_embeddings = self.rotary_embedding(hidden_states)
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask, rope_embeddings)
            hidden_states_list.append(hidden_states)
        if return_hidden_states:
            return hidden_states, tuple(hidden_states_list)
        return hidden_states

    def _pool(self, hidden_states, attention_mask=None):
        if self.pooling == "cls":
            return hidden_states[:, 0]
        return masked_mean(hidden_states, attention_mask)

    def _task_logits(self, pooled_output):
        hidden = tlx.relu(self.task_head_in(pooled_output))
        hidden = self.task_head_dropout(hidden)
        return self.task_head_out(hidden)

    def forward(self, input_ids, attention_mask=None, task: str | None = None):
        r"""Encode a batch and select the requested prediction head.

        Parameters
        ----------
        input_ids: ``Tensor``
            Integer model IDs of shape (B, L), in [0, vocab_size).
            The model consumes sequences, not edge_index or raw node features.
        attention_mask: ``Tensor``
            Shape (B, L); 1 marks visible tokens and 0 padding. None makes all
            positions visible. (default: :obj:`None`)
        task: str
            "mlm", "supervised", "pooled", or None. (default: :obj:`None`)

        Returns
        -------
        ``Tensor`` or dict
            "mlm": logits (B, L, vocab_size); "supervised": raw graph outputs
            (B, output_dim); "pooled": representations (B, hidden_size).
            None returns a dict with logits, mlm_logits, pooled_output, and
            last_hidden_state (B, L, hidden_size). No mode applies softmax.

        Raises
        ------
        ValueError
            If task is unsupported or L exceeds max_position_embeddings.

        Notes
        -----
        Classification aggregation belongs to the example trainer: softmax
        each serialization's logits, average probabilities by graph_id, then
        argmax. Regression averages predictions and reports raw-scale MAE.
        The trainer raises FloatingPointError on non-finite numerical results;
        this forward method itself does not perform those training checks.
        """
        if task not in {None, "mlm", "supervised", "pooled"}:
            raise ValueError(
                "task must be None, 'mlm', 'supervised', or 'pooled'.")
        hidden_states = self._embed(input_ids)
        rope_embeddings = self.rotary_embedding(hidden_states)
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask, rope_embeddings)
        if task == "mlm":
            return self.mlm_head(hidden_states)
        pooled_output = self._pool(hidden_states, attention_mask)
        if task == "pooled":
            return pooled_output
        if task == "supervised":
            return self._task_logits(pooled_output)
        mlm_logits = self.mlm_head(hidden_states)
        task_logits = self._task_logits(pooled_output)
        return {
            "logits": task_logits,
            "mlm_logits": mlm_logits,
            "pooled_output": pooled_output,
            "last_hidden_state": hidden_states,
        }

    def manifest(self):
        r"""Return architecture, parameter counts and initialization provenance.

        Returns
        -------
        dict
            The returned dict distinguishes random initialization from an official
            encoder load, including its revision, hash and coverage report when
            available. It does not identify a fitted tokenizer or downstream run.
        """
        manifest = {
            "encoder_type": self.encoder_type,
            "model_name": self.model_name,
            "pooling": self.pooling,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "max_position_embeddings": self.max_position_embeddings,
            "total_parameters": _parameter_count(self.all_weights),
            "trainable_parameters": _parameter_count(self.trainable_weights),
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.dropout_rate,
            "attention_probs_dropout_prob": self.attention_dropout_rate,
            "position_embedding_type": self.position_embedding_type,
            "layer_norm_eps": self.layer_norm_eps,
            "rope_theta": self.rope_theta,
            "rope_scaling": {
                "type": "ntk",
                "factor": self.rope_scaling_factor,
            },
            "framework": "tensorlayerx",
            "weight_source": "random_initialization",
            "official_checkpoint": None,
        }
        if self.initialization_manifest is not None:
            manifest.update(self.initialization_manifest)
            manifest["weight_source"] = self.initialization_manifest["mode"]
        else:
            manifest.update({
                "pretrained": False,
                "reproduction": False,
                "graph_embedding_init": "normal(stddev=0.02); zero PAD",
                "task_head_init": "normal(stddev=0.02); zero bias",
            })
        return manifest
