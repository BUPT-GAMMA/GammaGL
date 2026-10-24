"""BERT architecture and gradients against the pinned official implementation."""
import json
from pathlib import Path

import numpy as np
import pytest
import tensorlayerx as tlx

from gammagl.models import GraphBERT
from examples.graph_tokenizer.tlx_training import assign, gradient_context


REFERENCE = json.loads((Path(__file__).parent / 'fixtures/graph_bert_official.json').read_text())


def tiny_model(**kwargs):
    config = dict(vocab_size=11, output_dim=2, hidden_size=4, num_hidden_layers=1,
                  num_attention_heads=2, intermediate_size=6, max_position_embeddings=8,
                  dropout_rate=0., task_dropout=0.)
    config.update(kwargs)
    return GraphBERT(**config)


def gradients(loss, tape, weights, optimizer=None):
    if tlx.BACKEND == 'tensorflow':
        import tensorflow as tf
        return [tf.convert_to_tensor(value) for value in tape.gradient(loss, weights)]
    return (optimizer or tlx.optimizers.Adam(lr=0.)).gradient(loss, weights)


def official_weights(model):
    layer = model.encoder_layers[0]
    return {
        'embeddings.word_embeddings.weight': model.token_embeddings.embeddings,
        'embeddings.position_embeddings.weight': model.position_embeddings.embeddings,
        'embeddings.token_type_embeddings.weight': model.token_type_embeddings.embeddings,
        'embeddings.LayerNorm.weight': model.embedding_norm.gamma,
        'embeddings.LayerNorm.bias': model.embedding_norm.beta,
        'encoder.layer.0.attention.self.query.weight': layer.attention.q_proj.weights,
        'encoder.layer.0.attention.self.query.bias': layer.attention.q_proj.biases,
        'encoder.layer.0.attention.self.key.weight': layer.attention.k_proj.weights,
        'encoder.layer.0.attention.self.key.bias': layer.attention.k_proj.biases,
        'encoder.layer.0.attention.self.value.weight': layer.attention.v_proj.weights,
        'encoder.layer.0.attention.self.value.bias': layer.attention.v_proj.biases,
        'encoder.layer.0.attention.output.dense.weight': layer.attention.out_proj.weights,
        'encoder.layer.0.attention.output.dense.bias': layer.attention.out_proj.biases,
        'encoder.layer.0.attention.output.LayerNorm.weight': layer.attention_norm.gamma,
        'encoder.layer.0.attention.output.LayerNorm.bias': layer.attention_norm.beta,
        'encoder.layer.0.intermediate.dense.weight': layer.ffn_in.weights,
        'encoder.layer.0.intermediate.dense.bias': layer.ffn_in.biases,
        'encoder.layer.0.output.dense.weight': layer.ffn_out.weights,
        'encoder.layer.0.output.dense.bias': layer.ffn_out.biases,
        'encoder.layer.0.output.LayerNorm.weight': layer.ffn_norm.gamma,
        'encoder.layer.0.output.LayerNorm.bias': layer.ffn_norm.beta,
        'pooler.dense.weight': model.encoder_pooler.weights,
        'pooler.dense.bias': model.encoder_pooler.biases,
    }


def test_bert_forward_and_gradients_match_unmodified_official_model():
    model = tiny_model()
    model.set_eval()
    encoder = official_weights(model)
    mlm = dict(weight=model.mlm_head.weights, bias=model.mlm_head.biases)
    task = {'0.weight': model.task_head_in.weights, '0.bias': model.task_head_in.biases,
            '3.weight': model.task_head_out.weights, '3.bias': model.task_head_out.biases}
    for name, mapping in [('encoder', encoder), ('mlm', mlm), ('task', task)]:
        assert set(mapping) == set(REFERENCE[name])
        for key, weight in mapping.items():
            assert weight is not None, f'official {name}.{key} is missing'
            assign(weight, tlx.convert_to_tensor(REFERENCE[name][key], dtype=weight.dtype))
    ids = tlx.convert_to_tensor(REFERENCE['input_ids'], dtype=tlx.int64)
    attention = tlx.convert_to_tensor(REFERENCE['attention_mask'], dtype=tlx.int64)
    np.testing.assert_allclose(tlx.convert_to_numpy(model._embed(ids)), REFERENCE['embedding_output'], atol=2e-6)
    with gradient_context() as tape:
        output = model(ids, attention)
        labels = np.asarray(REFERENCE['labels']).reshape(-1)
        selected = np.flatnonzero(labels != -100)
        logits = tlx.gather(tlx.reshape(output['mlm_logits'], [-1, 11]), selected)
        loss = tlx.losses.softmax_cross_entropy_with_logits(
            logits, tlx.convert_to_tensor(labels[selected], dtype=tlx.int64))
        coefficients = tlx.reshape(tlx.arange(0, 32, dtype=tlx.float32), [2, 4, 4]) / 10.
        loss += tlx.reduce_sum(output['last_hidden_state'] * coefficients)
        loss += tlx.losses.softmax_cross_entropy_with_logits(
            output['logits'], tlx.convert_to_tensor([0, 1], dtype=tlx.int64))
    # The unused pooler has no gradient in either implementation.
    active = [(name, weight) for group, mapping in [('encoder', encoder), ('mlm', mlm), ('task', task)]
              for name, weight in mapping.items() if name in REFERENCE[group + '_gradients']]
    weights = [weight for _, weight in active]
    actual = gradients(loss, tape, weights)
    expected = [REFERENCE[group + '_gradients'][name]
                for group, mapping in [('encoder', encoder), ('mlm', mlm), ('task', task)]
                for name in mapping if name in REFERENCE[group + '_gradients']]
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(tlx.convert_to_numpy(result), reference, atol=3e-5, rtol=1e-4)
    for key, actual_key in [('hidden_states', 'last_hidden_state'), ('mlm_logits', 'mlm_logits'),
                            ('task_logits', 'logits')]:
        np.testing.assert_allclose(tlx.convert_to_numpy(output[actual_key]), REFERENCE[key], atol=2e-6)
    assert float(tlx.convert_to_numpy(loss)) == pytest.approx(REFERENCE['loss'], abs=3e-6)


def test_bert_mlm_bias_starts_zero_and_learns():
    model = tiny_model()
    bias = model.mlm_head.biases
    assert bias is not None
    np.testing.assert_array_equal(tlx.convert_to_numpy(bias), np.zeros(11))
    ids = tlx.convert_to_tensor([[3, 8, 4]], dtype=tlx.int64)
    optimizer = tlx.optimizers.Adam(lr=.01)
    with gradient_context() as tape:
        logits = model(ids, tlx.ones_like(ids), task='mlm')
        loss = tlx.losses.softmax_cross_entropy_with_logits(
            tlx.reshape(logits, [-1, 11]), tlx.convert_to_tensor([8, 8, 8], dtype=tlx.int64))
    gradient = gradients(loss, tape, [bias], optimizer)[0]
    assert np.isfinite(tlx.convert_to_numpy(gradient)).all()
    assert np.linalg.norm(tlx.convert_to_numpy(gradient)) > 0
    if tlx.BACKEND == 'torch':
        bias.grad = gradient
    optimizer.apply_gradients([(gradient, bias)])
    assert np.linalg.norm(tlx.convert_to_numpy(bias)) > 0


@pytest.mark.parametrize('pad', [0, 5])
def test_bert_pad_alias_is_zero_and_frozen_even_when_attention_is_one(pad):
    model = tiny_model(pad_token_id=pad)
    weight = model.token_embeddings.embeddings
    np.testing.assert_array_equal(tlx.convert_to_numpy(weight)[pad], np.zeros(4))
    ids = tlx.convert_to_tensor([[3, pad, 8, 4]], dtype=tlx.int64)
    before = tlx.convert_to_numpy(weight).copy()
    optimizer = tlx.optimizers.Adam(lr=.01)
    with gradient_context() as tape:
        logits = model(ids, tlx.ones_like(ids), task='mlm')
        loss = tlx.losses.softmax_cross_entropy_with_logits(
            tlx.reshape(logits, [-1, 11]), tlx.convert_to_tensor([8, 8, 8, 8], dtype=tlx.int64))
    gradient = gradients(loss, tape, [weight], optimizer)[0]
    np.testing.assert_array_equal(tlx.convert_to_numpy(gradient)[pad], np.zeros(4))
    assert np.linalg.norm(tlx.convert_to_numpy(gradient)[8]) > 0
    if tlx.BACKEND == 'torch':
        weight.grad = gradient
    optimizer.apply_gradients([(gradient, weight)])
    np.testing.assert_array_equal(tlx.convert_to_numpy(weight)[pad], np.zeros(4))
    assert not np.array_equal(tlx.convert_to_numpy(weight)[8], before[8])


def test_bert_uses_untruncated_normal_for_embeddings_and_linear_weights():
    tlx.set_seed(7)
    model = tiny_model(vocab_size=4096, hidden_size=64, num_attention_heads=4,
                       intermediate_size=256, max_position_embeddings=512)
    arrays = [model.token_embeddings.embeddings, model.position_embeddings.embeddings,
              model.encoder_layers[0].ffn_in.weights, model.mlm_head.weights,
              model.task_head_in.weights]
    for weight in arrays:
        values = tlx.convert_to_numpy(weight)
        assert values.std() == pytest.approx(.02, rel=.08)
        assert (np.abs(values) > .04).mean() > .02


@pytest.mark.parametrize('pad', [-1, 11])
def test_bert_rejects_pad_ids_outside_the_vocabulary(pad):
    with pytest.raises(ValueError, match='pad_token_id'):
        tiny_model(pad_token_id=pad)
