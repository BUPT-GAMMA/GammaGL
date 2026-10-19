"""Shared Transformer tensor contracts on the selected real TLX backend."""
import math

import numpy as np
import pytest
import tensorlayerx as tlx

pytestmark = pytest.mark.skipif(tlx.BACKEND not in ("torch", "tensorflow"),
                                reason="Shared Transformer gradient checks cover torch and TF eager")


def test_multihead_layout_and_merge_preserve_batch_and_token_order():
    from gammagl.models._graph_transformer import split_heads, merge_heads

    value = tlx.convert_to_tensor(np.arange(24).reshape(2, 3, 4), dtype=tlx.float32)
    heads = split_heads(value, num_attention_heads=2, head_size=2)
    np.testing.assert_array_equal(tlx.convert_to_numpy(heads), [
        [[[0, 1], [4, 5], [8, 9]], [[2, 3], [6, 7], [10, 11]]],
        [[[12, 13], [16, 17], [20, 21]], [[14, 15], [18, 19], [22, 23]]],
    ])
    np.testing.assert_array_equal(tlx.convert_to_numpy(merge_heads(heads, hidden_size=4)),
                                  np.arange(24).reshape(2, 3, 4))


def test_masked_pooling_and_attention_bias_include_empty_rows():
    from gammagl.models._graph_transformer import masked_mean, additive_attention_bias

    hidden = tlx.convert_to_tensor([[[1., 4.], [3., 8.], [9., 10.]],
                                    [[2., 6.], [4., 8.], [12., 16.]]])
    mask = tlx.convert_to_tensor([[1, 1, 0], [0, 0, 0]], dtype=tlx.int64)
    np.testing.assert_array_equal(tlx.convert_to_numpy(masked_mean(hidden, mask)), [[2, 6], [0, 0]])
    np.testing.assert_allclose(tlx.convert_to_numpy(masked_mean(hidden)), [[13 / 3, 22 / 3], [6, 10]])
    np.testing.assert_array_equal(tlx.convert_to_numpy(additive_attention_bias(mask)),
                                  [[[[0, 0, -10000]]], [[[-10000, -10000, -10000]]]])
    assert additive_attention_bias(None) is None


def test_scaled_attention_matches_closed_form_and_blocks_masked_value_gradients():
    from gammagl.models._graph_transformer import scaled_dot_product_attention

    query = tlx.convert_to_tensor([[[[math.sqrt(2), 0.]]]], dtype=tlx.float32)
    key = tlx.convert_to_tensor([[[[1., 0.], [0., 1.]]]])
    values = tlx.convert_to_tensor([[[[2., 4.], [6., 8.]]]])
    actual = scaled_dot_product_attention(query, key, values)
    expected = [(2 * math.e + 6) / (math.e + 1), (4 * math.e + 8) / (math.e + 1)]
    np.testing.assert_allclose(tlx.convert_to_numpy(actual), [[[expected]]], rtol=1e-6)
    mask = tlx.convert_to_tensor([[0, 1]], dtype=tlx.int64)
    actual = scaled_dot_product_attention(query, key, values, attention_mask=mask)
    np.testing.assert_array_equal(tlx.convert_to_numpy(actual), [[[[6, 8]]]])
    weights = tlx.nn.Linear(in_features=2, out_features=2, W_init="ones", b_init=None).weights

    def objective():
        return tlx.reduce_sum(scaled_dot_product_attention(
            query, key, tlx.reshape(weights, (1, 1, 2, 2)), attention_mask=mask))

    if tlx.BACKEND == "tensorflow":
        import tensorflow as tf
        with tf.GradientTape() as tape:
            loss = objective()
        gradient = tape.gradient(loss, weights)
    else:
        gradient = tlx.optimizers.Adam(lr=0.0).gradient(loss=objective(), weights=[weights])[0]
    np.testing.assert_array_equal(tlx.convert_to_numpy(gradient), [[0, 0], [1, 1]])


def test_attention_dropout_applies_to_probabilities_before_value_product():
    from gammagl.models._graph_transformer import scaled_dot_product_attention

    query = tlx.convert_to_tensor([[[[math.sqrt(2), 0.]]]], dtype=tlx.float32)
    key = tlx.convert_to_tensor([[[[1., 0.], [0., 1.]]]])
    values = tlx.convert_to_tensor([[[[2., 4.], [6., 8.]]]])
    probabilities = tlx.convert_to_tensor([[[[math.e / (math.e + 1), 1 / (math.e + 1)]]]],
                                          dtype=tlx.float32)
    dropout = tlx.nn.Dropout(p=0.5)
    dropout.set_train()
    tlx.set_seed(7)
    # Use the backend's real dropout mask on hand-derived probabilities.
    # The expected weighted value is independent of the attention implementation.
    dropped = tlx.convert_to_numpy(dropout(probabilities)).reshape(2)
    expected = dropped[0] * np.array([2., 4.]) + dropped[1] * np.array([6., 8.])
    tlx.set_seed(7)
    actual = scaled_dot_product_attention(query, key, values, dropout=dropout)
    np.testing.assert_allclose(tlx.convert_to_numpy(actual), [[[expected]]], rtol=1e-6)
    dropout.set_eval()
    actual = scaled_dot_product_attention(query, key, values, dropout=dropout)
    expected = [(2 * math.e + 6) / (math.e + 1), (4 * math.e + 8) / (math.e + 1)]
    np.testing.assert_allclose(tlx.convert_to_numpy(actual), [[[expected]]], rtol=1e-6)
