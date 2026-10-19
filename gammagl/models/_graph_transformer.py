"""Internal TLX tensor operations shared by the graph Transformer encoders.

Projection layers, position encodings and initialization remain encoder-specific.
These functions create no parameters or submodules.
"""

import math

import tensorlayerx as tlx


def _parameter_count(weights) -> int:
    total = 0
    for weight in weights:
        size = 1
        for dimension in tlx.get_tensor_shape(weight):
            size *= int(dimension)
        total += size
    return total


def masked_mean(hidden_states, attention_mask=None):
    """Pool (B, L, D) states with a (B, L) mask; empty rows return zeros."""
    if attention_mask is None:
        return tlx.reduce_mean(hidden_states, axis=1)
    mask = tlx.expand_dims(tlx.cast(attention_mask, tlx.float32), axis=-1)
    numerator = tlx.reduce_sum(hidden_states * mask, axis=1)
    denominator = tlx.reduce_sum(mask, axis=1)
    denominator = tlx.maximum(denominator, tlx.ones_like(denominator))
    return numerator / denominator


def additive_attention_bias(attention_mask):
    """Expand a (B, L) visibility mask to (B, 1, 1, L) additive key bias."""
    if attention_mask is None:
        return None
    mask = tlx.cast(attention_mask, tlx.float32)
    mask = tlx.expand_dims(tlx.expand_dims(mask, axis=1), axis=1)
    return (1.0 - mask) * -10000.0


def split_heads(value, num_attention_heads: int, head_size: int):
    """Reshape (B, L, H*D) projected states to (B, H, L, D)."""
    batch_size, sequence_length = tlx.get_tensor_shape(value)[:2]
    value = tlx.reshape(value, [batch_size, sequence_length, num_attention_heads, head_size])
    return tlx.transpose(value, [0, 2, 1, 3])


def merge_heads(value, hidden_size: int):
    """Merge (B, H, L, D) attention states into (B, L, H*D)."""
    value = tlx.transpose(value, [0, 2, 1, 3])
    batch_size, sequence_length = tlx.get_tensor_shape(value)[:2]
    return tlx.reshape(value, [batch_size, sequence_length, hidden_size])


def scaled_dot_product_attention(query, key, value, attention_mask=None, dropout=None):
    """Attend over (B, H, L, D) states, dropping probabilities after softmax.

    A (B, L) mask applies to keys. This retains the encoders' finite -10000
    masking convention, including its behavior for entirely masked key rows.
    ``dropout`` is the caller's existing layer, so its training mode is preserved.
    """
    head_size = int(tlx.get_tensor_shape(query)[-1])
    key_transposed = tlx.transpose(key, [0, 1, 3, 2])
    scores = tlx.matmul(query, key_transposed) / math.sqrt(head_size)
    bias = additive_attention_bias(attention_mask)
    if bias is not None:
        scores = scores + bias
    probabilities = tlx.softmax(scores, axis=-1)
    if dropout is not None:
        probabilities = dropout(probabilities)
    return tlx.matmul(probabilities, value)
