import json
from pathlib import Path

import numpy as np
import pytest
import tensorlayerx as tlx

from gammagl.models.graph_gte import GraphGTE
from gammagl.models.graph_gte_pretrained import _assign_tensor, load_pretrained_encoder
from gammagl.models import graph_gte_pretrained as pretrained


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("width", [2, 3], ids=["square", "rectangular"])
def test_assign_numpy_matrix_to_real_backend_variable(width, transpose):
    target = tlx.nn.Linear(in_features=width, out_features=2).weights
    expected = np.arange(2 * width, dtype=np.float32).reshape(2, width)
    source = expected.T.copy() if transpose else expected.copy()

    _assign_tensor(target, source, "linear.weight", transpose=transpose)

    np.testing.assert_array_equal(tlx.convert_to_numpy(target), expected)


def _checkpoint(model):
    layer = model.encoder_layers[0]
    # Official checkpoint dimensions are [out, in], independent of the target.
    entries = [
        ("new.embeddings.token_type_embeddings.weight", (1, 4),
         model.token_type_embeddings.embeddings),
        ("new.embeddings.LayerNorm.weight", (4,), model.embedding_norm.gamma),
        ("new.embeddings.LayerNorm.bias", (4,), model.embedding_norm.beta),
        ("new.encoder.layer.0.attention.qkv_proj.weight", (12, 4),
         layer.attention.qkv_proj.weights),
        ("new.encoder.layer.0.attention.qkv_proj.bias", (12,),
         layer.attention.qkv_proj.biases),
        ("new.encoder.layer.0.attention.o_proj.weight", (4, 4),
         layer.attention.out_proj.weights),
        ("new.encoder.layer.0.attention.o_proj.bias", (4,),
         layer.attention.out_proj.biases),
        ("new.encoder.layer.0.attn_ln.weight", (4,), layer.attention_norm.gamma),
        ("new.encoder.layer.0.attn_ln.bias", (4,), layer.attention_norm.beta),
        ("new.encoder.layer.0.mlp.up_gate_proj.weight", (12, 4),
         layer.mlp.up_gate_proj.weights),
        ("new.encoder.layer.0.mlp.down_proj.weight", (4, 6),
         layer.mlp.down_proj.weights),
        ("new.encoder.layer.0.mlp.down_proj.bias", (4,), layer.mlp.down_proj.biases),
        ("new.encoder.layer.0.mlp_ln.weight", (4,), layer.mlp_norm.gamma),
        ("new.encoder.layer.0.mlp_ln.bias", (4,), layer.mlp_norm.beta),
    ]
    tensors = {
        key: np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + index
        for index, (key, shape, _) in enumerate(entries)
    }
    tensors["new.embeddings.word_embeddings.weight"] = np.ones((7, 4), np.float32)
    tensors["classifier.weight"] = np.ones((2, 4), np.float32)
    tensors["classifier.bias"] = np.ones((2,), np.float32)
    return tensors, entries


def _tiny_model(vocab_size=16):
    return GraphGTE(vocab_size=vocab_size, output_dim=2, hidden_size=4,
                    num_hidden_layers=1, num_attention_heads=2,
                    intermediate_size=6, max_position_embeddings=8,
                    dropout_rate=0.0, task_dropout=0.0)


@pytest.mark.parametrize("vocab_size", [5, 7, 16], ids=["shrink", "same", "grow"])
def test_numpy_safetensors_converter_preserves_all_encoder_values(tmp_path, vocab_size):
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    model = _tiny_model(vocab_size)
    graph_embeddings = tlx.convert_to_numpy(model.token_embeddings.embeddings).copy()
    tensors, entries = _checkpoint(model)
    checkpoint = tmp_path / "encoder.safetensors"
    safetensors_numpy.save_file(tensors, str(checkpoint))

    report = load_pretrained_encoder(model, checkpoint)

    assert report["coverage"] == 1.0
    assert len(report["loaded"]) == len(entries) + 1
    assert report["missing"] == report["unexpected"] == report["shape_mismatches"] == []
    for key, _, target in entries:
        np.testing.assert_array_equal(tlx.convert_to_numpy(target), tensors[key])
    expected = graph_embeddings
    expected[:min(vocab_size, 7)] = tensors["new.embeddings.word_embeddings.weight"][:vocab_size]
    expected[model.pad_token_id] = 0
    np.testing.assert_array_equal(tlx.convert_to_numpy(model.token_embeddings.embeddings), expected)
    assert "new.embeddings.word_embeddings.weight" in report["loaded"]


def test_gte_pad_embedding_has_zero_value_and_gradient():
    model = _tiny_model()
    ids = tlx.convert_to_tensor([[0, 2, 3]], dtype=tlx.int64)
    coefficients = tlx.convert_to_tensor([1., 2., 3., 4.])
    if tlx.BACKEND == "tensorflow":
        import tensorflow as tf
        with tf.GradientTape() as tape:
            loss = tlx.reduce_sum(model._embed(ids) * coefficients)
        gradient = tf.convert_to_tensor(tape.gradient(loss, model.token_embeddings.embeddings))
    else:
        gradient = tlx.optimizers.Adam(lr=0.0).gradient(loss=tlx.reduce_sum(model._embed(ids) * coefficients),
                                                     weights=[model.token_embeddings.embeddings])[0]
    values = tlx.convert_to_numpy(model.token_embeddings.embeddings)
    gradient = tlx.convert_to_numpy(gradient)
    np.testing.assert_array_equal(values[0], np.zeros(4))
    np.testing.assert_array_equal(gradient[0], np.zeros(4))
    assert np.linalg.norm(gradient[2:4]) > 0


@pytest.mark.parametrize("problem", ["missing", "unexpected", "shape_mismatch"])
def test_numpy_safetensors_converter_rejects_incomplete_mapping(tmp_path, problem):
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    model = _tiny_model()
    tensors, _ = _checkpoint(model)
    key = "new.encoder.layer.0.attention.qkv_proj.weight"
    if problem == "missing":
        del tensors[key]
    elif problem == "unexpected":
        tensors["unknown.weight"] = np.zeros((1,), np.float32)
    else:
        tensors[key] = np.zeros((1, 1), np.float32)
    checkpoint = tmp_path / "bad_encoder.safetensors"
    safetensors_numpy.save_file(tensors, str(checkpoint))

    with pytest.raises(RuntimeError, match="Incomplete GTE encoder conversion"):
        load_pretrained_encoder(model, checkpoint)


GTE_REFERENCE = json.loads((Path(__file__).parent / "fixtures" /
                            "graph_gte_initialization_official.json").read_text())


@pytest.mark.parametrize("case", GTE_REFERENCE["cases"][:2], ids=["shrink", "grow"])
def test_pretrained_input_ids_match_official_embedding_hidden_states_and_gradients(tmp_path, monkeypatch, case):
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    checkpoint = tmp_path / "tiny.safetensors"
    safetensors_numpy.save_file({key: np.asarray(value, np.float32)
                                for key, value in GTE_REFERENCE["source"].items()}, str(checkpoint))
    monkeypatch.setattr(pretrained, "load_official_gte_config", lambda **_k: GTE_REFERENCE["config"])
    monkeypatch.setattr(pretrained, "download_official_gte_checkpoint", lambda **_k: (checkpoint, pretrained.sha256_file(checkpoint)))
    mapping = case["model_id_to_official_id"]
    model = GraphGTE.from_pretrained(case["vocab_size"], 2, pad_token_id=1,
                                     model_id_to_official_id=mapping)
    model.set_eval()
    inverse = np.argsort(mapping)
    ids = tlx.convert_to_tensor(inverse[np.asarray(case["input_ids"])], dtype=tlx.int64)
    attention = tlx.convert_to_tensor(case["attention_mask"], dtype=tlx.int64)
    np.testing.assert_allclose(tlx.convert_to_numpy(model._embed(ids)), case["embedding_output"], atol=2e-6)
    weights = [model.token_embeddings.embeddings, model.encoder_layers[0].attention.qkv_proj.weights]
    if tlx.BACKEND == "tensorflow":
        import tensorflow as tf
        with tf.GradientTape() as tape:
            hidden = model(ids, attention)["last_hidden_state"]
            loss = tlx.reduce_sum(hidden * tlx.reshape(tlx.arange(0, 32, dtype=tlx.float32), (2, 4, 4)) / 10.)
        gradients = [tf.convert_to_tensor(g) for g in tape.gradient(loss, weights)]
    else:
        hidden = model(ids, attention)["last_hidden_state"]
        loss = tlx.reduce_sum(hidden * tlx.reshape(tlx.arange(0, 32, dtype=tlx.float32), (2, 4, 4)) / 10.)
        gradients = tlx.optimizers.Adam(lr=0.0).gradient(loss, weights)
    np.testing.assert_allclose(tlx.convert_to_numpy(hidden), case["hidden_states"], atol=2e-6)
    np.testing.assert_allclose(tlx.convert_to_numpy(gradients[0]),
                               np.asarray(case["embedding_gradient"])[mapping], atol=3e-5, rtol=1e-4)
    np.testing.assert_allclose(tlx.convert_to_numpy(gradients[1]), case["qkv_gradient"], atol=3e-5, rtol=1e-4)
    assert model.pad_token_id == 1
    assert model.pretrained_manifest["reproduction"] is False


def test_vocabulary_permutation_includes_mlm_weights_and_bias():
    case = GTE_REFERENCE["cases"][0]
    model = _tiny_model(9)
    mapping = case["model_id_to_official_id"]
    embedding = tlx.convert_to_numpy(model.token_embeddings.embeddings).copy()
    decoder = np.asarray(case["mlm_weight"], np.float32)
    bias = np.asarray(case["mlm_bias"], np.float32) + np.arange(9, dtype=np.float32)
    _assign_tensor(model.mlm_head.weights, decoder, "decoder")
    _assign_tensor(model.mlm_head.biases, bias, "decoder bias")
    pretrained.remap_vocabulary(model, mapping)
    np.testing.assert_array_equal(tlx.convert_to_numpy(model.token_embeddings.embeddings), embedding[mapping])
    np.testing.assert_array_equal(tlx.convert_to_numpy(model.mlm_head.weights), decoder[mapping])
    np.testing.assert_array_equal(tlx.convert_to_numpy(model.mlm_head.biases), bias[mapping])
    logits = model.mlm_head(tlx.convert_to_tensor(case["hidden_states"], dtype=tlx.float32))
    expected = (np.asarray(case["mlm_logits"]) + np.arange(9))[..., mapping]
    np.testing.assert_allclose(tlx.convert_to_numpy(logits), expected, atol=1e-6)
    assert model.pad_token_id == 1


def test_official_random_config_does_not_load_checkpoint_and_uses_normal_initialization(monkeypatch):
    monkeypatch.setattr(pretrained, "load_official_gte_config", lambda **_k: GTE_REFERENCE["config"])
    monkeypatch.setattr(pretrained, "download_official_gte_checkpoint",
                        lambda **_k: pytest.fail("official random initialization loaded weights"))
    tlx.set_seed(7)
    model = GraphGTE.from_official_config(4096, 1, pad_token_id=0)
    assert model.hidden_size == 4 and model.num_hidden_layers == 1
    assert model.attention_dropout_rate == 0.0
    values = tlx.convert_to_numpy(model.token_embeddings.embeddings)
    np.testing.assert_array_equal(values[0], np.zeros(4))
    # An untruncated normal has tails outside 2 sigma (TF truncated normal has none).
    assert (np.abs(values[1:]) > 0.04).sum() > 100
    assert model.initialization_manifest["mode"] == "official_random"
    assert model.initialization_manifest["pretrained"] is False


@pytest.mark.parametrize("mapping", [[0] * 9, list(range(8)), [False] + list(range(1, 9))])
def test_invalid_vocabulary_mapping_does_not_modify_model(mapping):
    model = _tiny_model(9)
    before = [tlx.convert_to_numpy(w).copy() for w in model.all_weights]
    with pytest.raises(ValueError, match="bijection"):
        pretrained.remap_vocabulary(model, mapping)
    for weight, expected in zip(model.all_weights, before):
        np.testing.assert_array_equal(tlx.convert_to_numpy(weight), expected)
