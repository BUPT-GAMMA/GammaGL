"""Native TensorLayerX BERT encoder for serialized graph tokens."""

from __future__ import annotations

import math

import tensorlayerx as tlx


def _truncated_normal():
    return tlx.initializers.TruncatedNormal(stddev=0.02)


def _linear(in_features: int, out_features: int, bias: bool = True):
    return tlx.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        W_init=_truncated_normal(),
        b_init=tlx.initializers.Constant(0.0) if bias else None,
    )


def _parameter_count(weights) -> int:
    total = 0
    for weight in weights:
        size = 1
        for dimension in tlx.get_tensor_shape(weight):
            size *= int(dimension)
        total += size
    return total


def masked_mean(hidden_states, attention_mask=None):
    if attention_mask is None:
        return tlx.reduce_mean(hidden_states, axis=1)
    mask = tlx.expand_dims(tlx.cast(attention_mask, tlx.float32), axis=-1)
    numerator = tlx.reduce_sum(hidden_states * mask, axis=1)
    denominator = tlx.reduce_sum(mask, axis=1)
    denominator = tlx.maximum(denominator, tlx.ones_like(denominator))
    return numerator / denominator


def additive_attention_bias(attention_mask):
    if attention_mask is None:
        return None
    mask = tlx.cast(attention_mask, tlx.float32)
    mask = tlx.expand_dims(tlx.expand_dims(mask, axis=1), axis=1)
    return (1.0 - mask) * -10000.0


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

    def forward(self, hidden_states, attention_mask=None):
        query = self._split_heads(self.q_proj(hidden_states))
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))
        key_transposed = tlx.transpose(key, [0, 1, 3, 2])
        scores = tlx.matmul(query, key_transposed) / math.sqrt(self.head_size)
        bias = additive_attention_bias(attention_mask)
        if bias is not None:
            scores = scores + bias
        probabilities = tlx.softmax(scores, axis=-1)
        probabilities = self.attention_dropout(probabilities)
        context = tlx.matmul(probabilities, value)
        return self.out_proj(self._merge_heads(context))


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
    """BERT-Small encoder with MLM and graph-level task heads."""

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
    """Strict BERT-Small GraphTokenizer encoder."""
