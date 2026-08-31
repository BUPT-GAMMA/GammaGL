"""Native TensorLayerX GTE encoder for serialized graph tokens."""

from __future__ import annotations

import math

import tensorlayerx as tlx

try:
    from .graph_bert import (
        _linear,
        _parameter_count,
        _truncated_normal,
        additive_attention_bias,
        masked_mean,
    )
except ImportError:  # Allows direct file loading in focused tests.
    from graph_bert import (
        _linear,
        _parameter_count,
        _truncated_normal,
        additive_attention_bias,
        masked_mean,
    )


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

    def _split_heads(self, value):
        batch_size, sequence_length = tlx.get_tensor_shape(value)[:2]
        value = tlx.reshape(
            value,
            [batch_size, sequence_length, self.num_attention_heads, self.head_size],
        )
        return tlx.transpose(value, [0, 2, 1, 3])

    def _merge_heads(self, value):
        value = tlx.transpose(value, [0, 2, 1, 3])
        batch_size, sequence_length = tlx.get_tensor_shape(value)[:2]
        return tlx.reshape(value, [batch_size, sequence_length, self.hidden_size])

    def forward(self, hidden_states, attention_mask=None, rope_embeddings=None):
        if rope_embeddings is None:
            raise ValueError("GTE attention requires rotary embeddings.")
        query, key, value = tlx.split(
            self.qkv_proj(hidden_states), 3, axis=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        query, key = apply_rotary_pos_emb(
            query, key, rope_embeddings[0], rope_embeddings[1])
        key_transposed = tlx.transpose(key, [0, 1, 3, 2])
        scores = tlx.matmul(query, key_transposed) / math.sqrt(self.head_size)
        bias = additive_attention_bias(attention_mask)
        if bias is not None:
            scores = scores + bias
        probabilities = self.attention_dropout(tlx.softmax(scores, axis=-1))
        context = tlx.matmul(probabilities, value)
        return self.out_proj(self._merge_heads(context))


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
    """Native TLX GTE encoder; ``from_pretrained`` loads its GTE encoder."""

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
        self.mlm_head = _linear(self.hidden_size, self.vocab_size, bias=False)
        task_hidden_size = max(1, self.hidden_size // 2)
        self.task_head_in = _linear(self.hidden_size, task_hidden_size)
        self.task_head_dropout = tlx.nn.Dropout(p=self.task_dropout_rate)
        self.task_head_out = _linear(task_hidden_size, self.output_dim)
        self.pretrained_manifest = None

    @classmethod
    def from_pretrained(cls, vocab_size: int, output_dim: int, cache_dir=None, **kwargs):
        """Build native TLX GraphGTE with official encoder weights only."""
        from .graph_gte_pretrained import (
            GTE_CHECKPOINT, GTE_MODEL_ID, GTE_REVISION,
            download_official_gte_checkpoint, load_official_gte_config,
            load_pretrained_encoder,
        )
        config = load_official_gte_config(cache_dir=cache_dir)
        architecture = {
            "hidden_size": config["hidden_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            "intermediate_size": config["intermediate_size"],
            "max_position_embeddings": config["max_position_embeddings"],
            "dropout_rate": config["hidden_dropout_prob"],
            "attention_dropout_rate": config["attention_probs_dropout_prob"],
            "layer_norm_eps": config["layer_norm_eps"],
            "pad_token_id": config["pad_token_id"],
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
        model = cls(vocab_size=vocab_size, output_dim=output_dim, **architecture)
        path, actual_hash = download_official_gte_checkpoint(cache_dir=cache_dir)
        report = load_pretrained_encoder(model, path)
        model.pretrained_manifest = {
            "model_id": GTE_MODEL_ID, "revision": GTE_REVISION,
            "checkpoint_files": [GTE_CHECKPOINT], "checkpoint_sha256": {GTE_CHECKPOINT: actual_hash},
            "pretrained": True,
            "encoder_expected_tensors": len(report["expected"]),
            "encoder_loaded_tensors": len(report["loaded"]),
            "encoder_coverage": report["coverage"],
            "missing": report["missing"], "unexpected": report["unexpected"],
            "ignored": report["ignored"],
            "ignored_checkpoint_keys": report["ignored"],
            "shape_mismatches": report["shape_mismatches"],
            "graph_embedding_init": "new_truncated_normal(stddev=0.02)",
            "task_head_init": "new_truncated_normal(stddev=0.02)",
            "reproduction": True,
        }
        return model

    def _embed(self, input_ids):
        sequence_length = int(tlx.get_tensor_shape(input_ids)[1])
        if sequence_length > self.max_position_embeddings:
            raise ValueError("Input sequence length exceeds max_position_embeddings.")
        word_embeddings = self.token_embeddings(input_ids)
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
        """Run the native encoder from raw embeddings for reference testing."""
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
        if task not in {None, "mlm", "supervised"}:
            raise ValueError("task must be None, 'mlm', or 'supervised'.")
        hidden_states = self._embed(input_ids)
        rope_embeddings = self.rotary_embedding(hidden_states)
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask, rope_embeddings)
        if task == "mlm":
            return self.mlm_head(hidden_states)
        pooled_output = self._pool(hidden_states, attention_mask)
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
        if self.pretrained_manifest is not None:
            manifest.update(self.pretrained_manifest)
            manifest["weight_source"] = "official_gte_encoder"
        else:
            manifest.update({
                "pretrained": False,
                "reproduction": False,
                "graph_embedding_init": "new_truncated_normal(stddev=0.02)",
                "task_head_init": "new_truncated_normal(stddev=0.02)",
            })
        return manifest
