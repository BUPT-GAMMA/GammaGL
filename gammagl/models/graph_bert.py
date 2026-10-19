"""Native TensorLayerX BERT encoder for serialized graph tokens."""

from __future__ import annotations

import tensorlayerx as tlx

from ._graph_transformer import (
    _parameter_count,
    masked_mean,
    merge_heads,
    scaled_dot_product_attention,
    split_heads,
)


def _truncated_normal():
    return tlx.initializers.TruncatedNormal(stddev=0.02)


def _linear(in_features: int, out_features: int, bias: bool = True):
    return tlx.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        W_init=_truncated_normal(),
        b_init=tlx.initializers.Constant(0.0) if bias else None,
    )


class GraphTokenSelfAttention(tlx.nn.Module):
    """Backend-neutral scaled dot-product multi-head self-attention."""

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
        self.q_proj = _linear(self.hidden_size, self.hidden_size)
        self.k_proj = _linear(self.hidden_size, self.hidden_size)
        self.v_proj = _linear(self.hidden_size, self.hidden_size)
        self.out_proj = _linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = tlx.nn.Dropout(p=float(attention_dropout_rate))

    def forward(self, hidden_states, attention_mask=None):
        query = split_heads(self.q_proj(hidden_states), self.num_attention_heads, self.head_size)
        key = split_heads(self.k_proj(hidden_states), self.num_attention_heads, self.head_size)
        value = split_heads(self.v_proj(hidden_states), self.num_attention_heads, self.head_size)
        context = scaled_dot_product_attention(
            query, key, value, attention_mask, dropout=self.attention_dropout)
        return self.out_proj(merge_heads(context, self.hidden_size))


class GraphTokenTransformerLayer(tlx.nn.Module):
    """Post-normalization BERT encoder block."""

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
        self.attention = GraphTokenSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout_rate=attention_dropout_rate,
        )
        self.attention_norm = tlx.nn.LayerNorm(
            normalized_shape=hidden_size,
            epsilon=layer_norm_eps,
            gamma_init="ones",
            beta_init="zeros",
        )
        self.ffn_in = _linear(hidden_size, intermediate_size)
        self.ffn_out = _linear(intermediate_size, hidden_size)
        self.ffn_norm = tlx.nn.LayerNorm(
            normalized_shape=hidden_size,
            epsilon=layer_norm_eps,
            gamma_init="ones",
            beta_init="zeros",
        )
        self.hidden_dropout = tlx.nn.Dropout(p=float(dropout_rate))

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(
            hidden_states, attention_mask=attention_mask)
        hidden_states = self.attention_norm(
            hidden_states + self.hidden_dropout(attention_output))
        ffn_output = self.ffn_out(tlx.gelu(self.ffn_in(hidden_states)))
        return self.ffn_norm(hidden_states + self.hidden_dropout(ffn_output))


class GraphTokenTransformer(tlx.nn.Module):
    r"""BERT-Small encoder for graph token sequences, with MLM and graph task heads.

    Implements the BERT encoder used in `"Graph Tokenization for Bridging
    Graphs and Transformers" <https://arxiv.org/abs/2603.11099>`_. See the
    :doc:`GraphTokenizer guide </notes/graph_tokenizer>` for the pinned
    official implementation and the supported TLX torch / TensorFlow eager
    training paths. This constructor initializes weights randomly.

    Parameters
    ----------
    vocab_size: int
        Number of model token IDs. Must exceed every input ID; use
        :meth:`gammagl.transforms.GraphTokenizer.validate_model_vocab`.
    output_dim: int
        Number of graph targets, or number of classes for classification.
    hidden_size: int
        Token embedding and encoder width. (default: ``512``)
    num_hidden_layers: int
        Number of encoder blocks. (default: ``4``)
    num_attention_heads: int
        Attention heads; must divide hidden_size. (default: ``4``)
    intermediate_size: int
        Feed-forward hidden width. (default: ``2048``)
    max_position_embeddings: int
        Maximum input length including special tokens. This API default is
        distinct from the trainer's 8096-token preset. (default: ``768``)
    dropout_rate: float
        Embedding and encoder hidden dropout. (default: ``0.1``)
    attention_dropout_rate: float
        Attention probability dropout; None uses dropout_rate.
        (default: :obj:`None`)
    layer_norm_eps: float
        Layer normalization epsilon. (default: ``1e-12``)
    pad_token_id: int
        Padding ID recorded in model metadata. Supply attention_mask to
        exclude padding from attention and mean pooling. (default: ``0``)
    type_vocab_size: int
        Token-type embedding vocabulary; all inputs use type 0.
        (default: ``2``)
    task_type: str
        Task metadata, usually "regression" or "classification"; it does
        not select a loss or apply softmax. (default: ``"regression"``)
    pooling: str
        "mean" for masked mean pooling or "cls" for the first token.
        (default: ``"mean"``)
    task_dropout: float
        Graph task head dropout. (default: ``0.1``)
    name: str
        TensorLayerX module name. (default: :obj:`None`)

    Raises
    ------
    ValueError
        If pooling is unsupported, position capacity is nonpositive, or
        hidden_size is not divisible by the attention head count.

    Examples
    --------
    >>> import tensorlayerx as tlx
    >>> from gammagl.models import GraphTokenTransformer
    >>> model = GraphTokenTransformer(32, 2, hidden_size=16,
    ...     num_hidden_layers=1, num_attention_heads=4, intermediate_size=32)
    >>> model.set_eval()
    >>> ids = tlx.convert_to_tensor([[3, 8, 4]], dtype=tlx.int64)
    >>> logits = model(ids, tlx.ones_like(ids), task="supervised")
    >>> tuple(logits.shape)
    (1, 2)
    """

    encoder_type = "bert"
    model_name = "bert-small"
    position_embedding_type = "absolute"
    hidden_act = "gelu"

    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
        hidden_size: int = 512,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 768,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float | None = None,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        type_vocab_size: int = 2,
        task_type: str = "regression",
        pooling: str = "mean",
        task_dropout: float = 0.1,
        name: str | None = None,
    ):
        super().__init__(name=name)
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'.")
        if max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive.")
        attention_dropout_rate = (
            dropout_rate if attention_dropout_rate is None
            else attention_dropout_rate
        )
        self.vocab_size = int(vocab_size)
        self.output_dim = int(output_dim)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size)
        self.max_position_embeddings = int(max_position_embeddings)
        self.dropout_rate = float(dropout_rate)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.layer_norm_eps = float(layer_norm_eps)
        self.pad_token_id = int(pad_token_id)
        self.type_vocab_size = int(type_vocab_size)
        self.task_type = str(task_type)
        self.pooling = pooling
        self.task_dropout_rate = float(task_dropout)

        self.token_embeddings = tlx.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            E_init=_truncated_normal(),
        )
        self.position_embeddings = tlx.nn.Embedding(
            num_embeddings=self.max_position_embeddings,
            embedding_dim=self.hidden_size,
            E_init=_truncated_normal(),
        )
        self.token_type_embeddings = tlx.nn.Embedding(
            num_embeddings=self.type_vocab_size,
            embedding_dim=self.hidden_size,
            E_init=_truncated_normal(),
        )
        self.embedding_norm = tlx.nn.LayerNorm(
            normalized_shape=self.hidden_size,
            epsilon=self.layer_norm_eps,
            gamma_init="ones",
            beta_init="zeros",
        )
        self.embedding_dropout = tlx.nn.Dropout(p=self.dropout_rate)
        self.encoder_layers = tlx.nn.ModuleList([
            GraphTokenTransformerLayer(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                layer_norm_eps=self.layer_norm_eps,
            )
            for _ in range(self.num_hidden_layers)
        ])
        # Retain the standard BERT pooler parameters used by the canonical architecture.
        self.encoder_pooler = _linear(self.hidden_size, self.hidden_size)
        self.mlm_head = _linear(self.hidden_size, self.vocab_size, bias=False)
        task_hidden_size = max(1, self.hidden_size // 2)
        self.task_head_in = _linear(self.hidden_size, task_hidden_size)
        self.task_head_dropout = tlx.nn.Dropout(p=self.task_dropout_rate)
        self.task_head_out = _linear(task_hidden_size, self.output_dim)

    def _embed(self, input_ids):
        shape = tlx.get_tensor_shape(input_ids)
        sequence_length = int(shape[1])
        if sequence_length > self.max_position_embeddings:
            raise ValueError("Input sequence length exceeds max_position_embeddings.")
        word_embeddings = self.token_embeddings(input_ids)
        positions = tlx.cumsum(tlx.ones_like(input_ids), axis=1) - 1
        position_embeddings = self.position_embeddings(positions)
        token_type_ids = input_ids * 0
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        hidden_states = (
            word_embeddings + position_embeddings + token_type_embeddings)
        return self.embedding_dropout(self.embedding_norm(hidden_states))

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
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
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
        r"""Return architecture, framework and parameter-count metadata.

        Returns
        -------
        dict
            Counts refer to the model's current all_weights and trainable_weights.
            This manifest does not bind a tokenizer or validate checkpoint compatibility.
        """
        return {
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
            "framework": "tensorlayerx",
        }


class GraphBERT(GraphTokenTransformer):
    r"""BERT-Small graph token encoder from the `GraphTokenizer paper
    <https://arxiv.org/abs/2603.11099>`_.

    Uses learned absolute positions, GELU blocks and randomly initialized MLM
    and graph task heads. This constructor does not load a text BERT checkpoint.
    The native TLX implementation supports torch and TensorFlow eager.

    Parameters
    ----------
    vocab_size: int
        Number of model token IDs. Must exceed every input ID; use
        :meth:`gammagl.transforms.GraphTokenizer.validate_model_vocab`.
    output_dim: int
        Number of graph targets, or number of classes for classification.
    hidden_size: int
        Token embedding and encoder width. (default: ``512``)
    num_hidden_layers: int
        Number of encoder blocks. (default: ``4``)
    num_attention_heads: int
        Attention heads; must divide hidden_size. (default: ``4``)
    intermediate_size: int
        Feed-forward hidden width. (default: ``2048``)
    max_position_embeddings: int
        Maximum input length including special tokens. This API default is
        distinct from the trainer's 8096-token preset. (default: ``768``)
    dropout_rate: float
        Embedding and encoder hidden dropout. (default: ``0.1``)
    attention_dropout_rate: float
        Attention probability dropout; None uses dropout_rate.
        (default: :obj:`None`)
    layer_norm_eps: float
        Layer normalization epsilon. (default: ``1e-12``)
    pad_token_id: int
        Padding ID recorded in model metadata. Supply attention_mask to
        exclude padding from attention and mean pooling. (default: ``0``)
    type_vocab_size: int
        Token-type embedding vocabulary; all inputs use type 0.
        (default: ``2``)
    task_type: str
        Task metadata, usually "regression" or "classification"; it does
        not select a loss or apply softmax. (default: ``"regression"``)
    pooling: str
        "mean" for masked mean pooling or "cls" for the first token.
        (default: ``"mean"``)
    task_dropout: float
        Graph task head dropout. (default: ``0.1``)
    name: str
        TensorLayerX module name. (default: :obj:`None`)

    Raises
    ------
    ValueError
        If pooling is unsupported, position capacity is nonpositive, or
        hidden_size is not divisible by the attention head count.

    Notes
    -----
    Inputs are integer model IDs of shape (B, L), with an optional visibility
    mask of the same shape. Raw graph features must first pass through
    :class:`~gammagl.transforms.GraphTokenizer`. The inherited :meth:`forward`
    returns MLM logits (B, L, vocab_size), graph outputs (B, output_dim), pooled
    features (B, hidden_size), or a dict of all outputs according to task.
    Outputs are raw scores; classification softmax and graph-wise aggregation
    belong to the trainer. See :doc:`/notes/graph_tokenizer` for the protocol.

    Examples
    --------
    >>> import tensorlayerx as tlx
    >>> from gammagl.models import GraphBERT
    >>> model = GraphBERT(32, 2, hidden_size=16, num_hidden_layers=1,
    ...     num_attention_heads=4, intermediate_size=32)
    >>> model.set_eval()
    >>> ids = tlx.convert_to_tensor([[3, 8, 4]], dtype=tlx.int64)
    >>> tuple(model(ids, tlx.ones_like(ids), task="supervised").shape)
    (1, 2)
    """
