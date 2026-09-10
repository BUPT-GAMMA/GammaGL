import pytest
import tensorlayerx as tlx

from gammagl.models.graph_gte import GraphGTE
from gammagl.models.graph_gte_pretrained import load_pretrained_encoder


torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def _tiny_checkpoint_tensors(model):
    layer = model.encoder_layers[0]
    return {
        "new.embeddings.token_type_embeddings.weight": torch.ones_like(
            model.token_type_embeddings.embeddings),
        "new.embeddings.LayerNorm.weight": torch.ones_like(model.embedding_norm.gamma),
        "new.embeddings.LayerNorm.bias": torch.ones_like(model.embedding_norm.beta),
        "new.encoder.layer.0.attention.qkv_proj.weight": torch.ones_like(
            layer.attention.qkv_proj.weights),
        "new.encoder.layer.0.attention.qkv_proj.bias": torch.ones_like(
            layer.attention.qkv_proj.biases),
        "new.encoder.layer.0.attention.o_proj.weight": torch.ones_like(
            layer.attention.out_proj.weights),
        "new.encoder.layer.0.attention.o_proj.bias": torch.ones_like(
            layer.attention.out_proj.biases),
        "new.encoder.layer.0.attn_ln.weight": torch.ones_like(layer.attention_norm.gamma),
        "new.encoder.layer.0.attn_ln.bias": torch.ones_like(layer.attention_norm.beta),
        "new.encoder.layer.0.mlp.up_gate_proj.weight": torch.ones_like(
            layer.mlp.up_gate_proj.weights),
        "new.encoder.layer.0.mlp.down_proj.weight": torch.ones_like(
            layer.mlp.down_proj.weights),
        "new.encoder.layer.0.mlp.down_proj.bias": torch.ones_like(
            layer.mlp.down_proj.biases),
        "new.encoder.layer.0.mlp_ln.weight": torch.ones_like(layer.mlp_norm.gamma),
        "new.encoder.layer.0.mlp_ln.bias": torch.ones_like(layer.mlp_norm.beta),
    }


def test_graph_gte_loads_a_local_checkpoint_and_runs_forward(tmp_path):
    model = GraphGTE(
        vocab_size=8, output_dim=2, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=4, intermediate_size=32, max_position_embeddings=16,
    )
    checkpoint = tmp_path / "gte.safetensors"
    safetensors_torch.save_file(_tiny_checkpoint_tensors(model), str(checkpoint))

    report = load_pretrained_encoder(model, checkpoint)
    logits = model(
        tlx.convert_to_tensor([[1, 2, 3]], dtype=tlx.int64),
        attention_mask=tlx.convert_to_tensor([[1, 1, 1]], dtype=tlx.int64),
        task="supervised",
    )

    assert report["coverage"] == 1.0
    assert tlx.get_tensor_shape(logits) == [1, 2]
