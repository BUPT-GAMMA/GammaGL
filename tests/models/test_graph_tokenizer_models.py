import hashlib

import numpy as np
import pytest
import tensorlayerx as tlx

if tlx.BACKEND != "torch":
    pytest.skip("Native torch model checks require TLX torch", allow_module_level=True)
torch = pytest.importorskip("torch")

from gammagl.models.graph_bert import GraphBERT
from gammagl.models.graph_gte import GraphGTE, apply_rotary_pos_emb
from gammagl.models import graph_gte_pretrained


MODEL_CLASSES = (GraphBERT, GraphGTE)


def _tiny_model(model_class, output_dim=2):
    return model_class(
        vocab_size=32,
        output_dim=output_dim,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
        dropout_rate=0.0,
        task_dropout=0.0,
    )


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=("bert", "gte"))
def test_native_models_initialize_and_return_requested_output_shapes(model_class):
    model = _tiny_model(model_class, output_dim=3)
    input_ids = torch.tensor([[3, 5, 6, 4], [3, 7, 0, 0]], dtype=torch.long)
    attention = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.long)

    supervised = model(input_ids, attention, task="supervised")
    mlm = model(input_ids, attention, task="mlm")

    assert tuple(supervised.shape) == (2, 3)
    assert tuple(mlm.shape) == (2, 4, 32)
    assert model.manifest()["framework"] == "tensorlayerx"


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=("bert", "gte"))
def test_attention_mask_makes_padding_tokens_inert(model_class):
    model = _tiny_model(model_class)
    model.set_eval()
    attention = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    padded = torch.tensor([[3, 5, 4, 0]], dtype=torch.long)
    changed_padding = torch.tensor([[3, 5, 4, 17]], dtype=torch.long)

    first = model(padded, attention, task="supervised")
    second = model(changed_padding, attention, task="supervised")

    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=("bert", "gte"))
def test_native_models_backpropagate_finite_gradients(model_class):
    model = _tiny_model(model_class)
    input_ids = torch.tensor([[3, 5, 6, 4]], dtype=torch.long)
    attention = torch.ones_like(input_ids)

    loss = (model(input_ids, attention, task="mlm").sum()
            + model(input_ids, attention, task="supervised").sum())
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_graph_bert_public_default_architecture_contract():
    manifest = GraphBERT(vocab_size=32, output_dim=1).manifest()

    assert {key: manifest[key] for key in (
        "encoder_type", "model_name", "hidden_size", "num_hidden_layers",
        "num_attention_heads", "max_position_embeddings",
    )} == {
        "encoder_type": "bert", "model_name": "bert-small", "hidden_size": 512,
        "num_hidden_layers": 4, "num_attention_heads": 4, "max_position_embeddings": 768,
    }


def test_graph_gte_uses_rope_without_absolute_position_embeddings():
    model = _tiny_model(GraphGTE)
    query = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    key = torch.tensor([[[[5.0, 6.0, 7.0, 8.0]]]])
    cosine = torch.zeros_like(query)
    sine = torch.ones_like(query)

    rotated_query, rotated_key = apply_rotary_pos_emb(query, key, cosine, sine)

    assert not hasattr(model, "position_embeddings")
    assert model.manifest()["position_embedding_type"] == "rope"
    assert torch.equal(rotated_query, torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]]))
    assert torch.equal(rotated_key, torch.tensor([[[[-7.0, -8.0, 5.0, 6.0]]]]))


def _synthetic_checkpoint(model):
    layer = model.encoder_layers[0]
    tensors = {
        "new.embeddings.token_type_embeddings.weight": model.token_type_embeddings.embeddings,
        "new.embeddings.LayerNorm.weight": model.embedding_norm.gamma,
        "new.embeddings.LayerNorm.bias": model.embedding_norm.beta,
        "new.encoder.layer.0.attention.qkv_proj.weight": layer.attention.qkv_proj.weights,
        "new.encoder.layer.0.attention.qkv_proj.bias": layer.attention.qkv_proj.biases,
        "new.encoder.layer.0.attention.o_proj.weight": layer.attention.out_proj.weights,
        "new.encoder.layer.0.attention.o_proj.bias": layer.attention.out_proj.biases,
        "new.encoder.layer.0.attn_ln.weight": layer.attention_norm.gamma,
        "new.encoder.layer.0.attn_ln.bias": layer.attention_norm.beta,
        "new.encoder.layer.0.mlp.up_gate_proj.weight": layer.mlp.up_gate_proj.weights,
        "new.encoder.layer.0.mlp.down_proj.weight": layer.mlp.down_proj.weights,
        "new.encoder.layer.0.mlp.down_proj.bias": layer.mlp.down_proj.biases,
        "new.encoder.layer.0.mlp_ln.weight": layer.mlp_norm.gamma,
        "new.encoder.layer.0.mlp_ln.bias": layer.mlp_norm.beta,
        "new.embeddings.word_embeddings.weight": torch.zeros((7, 16)),
        "classifier.weight": torch.zeros((2, 16)),
        "classifier.bias": torch.zeros((2,)),
    }
    return {
        key: torch.arange(value.numel(), dtype=torch.float32).reshape(value.shape)
        if key.startswith("new.encoder") else torch.ones_like(value.detach())
        for key, value in tensors.items()
    }


def test_converter_loads_synthetic_checkpoint_with_full_coverage(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    source = _tiny_model(GraphGTE)
    tensors = _synthetic_checkpoint(source)
    checkpoint = tmp_path / "tiny.safetensors"
    safetensors_torch.save_file(tensors, str(checkpoint))
    target = _tiny_model(GraphGTE)

    report = graph_gte_pretrained.load_pretrained_encoder(target, checkpoint)

    assert report["coverage"] == 1.0
    assert report["missing"] == report["shape_mismatches"] == report["unexpected"] == []
    assert "new.embeddings.word_embeddings.weight" in report["loaded"]
    assert "new.embeddings.word_embeddings.weight" not in report["ignored"]
    np.testing.assert_array_equal(
        tlx.convert_to_numpy(target.encoder_layers[0].attention.qkv_proj.weights),
        tensors["new.encoder.layer.0.attention.qkv_proj.weight"].numpy())


def test_converter_rejects_synthetic_shape_mismatch(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    model = _tiny_model(GraphGTE)
    tensors = _synthetic_checkpoint(model)
    tensors["new.encoder.layer.0.attention.qkv_proj.weight"] = torch.zeros((1, 1))
    checkpoint = tmp_path / "bad.safetensors"
    safetensors_torch.save_file(tensors, str(checkpoint))

    with pytest.raises(RuntimeError, match="Incomplete GTE encoder conversion"):
        graph_gte_pretrained.load_pretrained_encoder(model, checkpoint)


def test_checksum_helper_hashes_actual_file_bytes(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"synthetic checkpoint")

    assert graph_gte_pretrained.sha256_file(checkpoint) == hashlib.sha256(
        b"synthetic checkpoint").hexdigest()
