import numpy as np
import pytest
import tensorlayerx as tlx

from gammagl.models.graph_gte import GraphGTE
from gammagl.models.graph_gte_pretrained import _assign_tensor, load_pretrained_encoder


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


def _tiny_model():
    return GraphGTE(vocab_size=16, output_dim=2, hidden_size=4,
                    num_hidden_layers=1, num_attention_heads=2,
                    intermediate_size=6, max_position_embeddings=8,
                    dropout_rate=0.0, task_dropout=0.0)


def test_numpy_safetensors_converter_preserves_all_encoder_values(tmp_path):
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    model = _tiny_model()
    graph_embeddings = tlx.convert_to_numpy(model.token_embeddings.embeddings).copy()
    tensors, entries = _checkpoint(model)
    checkpoint = tmp_path / "encoder.safetensors"
    safetensors_numpy.save_file(tensors, str(checkpoint))

    report = load_pretrained_encoder(model, checkpoint)

    assert report["coverage"] == 1.0
    assert len(report["loaded"]) == len(entries)
    assert report["missing"] == report["unexpected"] == report["shape_mismatches"] == []
    for key, _, target in entries:
        np.testing.assert_array_equal(tlx.convert_to_numpy(target), tensors[key])
    np.testing.assert_array_equal(
        tlx.convert_to_numpy(model.token_embeddings.embeddings), graph_embeddings)


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
