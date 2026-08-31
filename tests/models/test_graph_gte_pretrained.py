import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorlayerx as tlx


torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


ROOT = Path(__file__).resolve().parents[2]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _modules():
    _load("graph_bert", "gammagl/models/graph_bert.py")
    pretrained = _load("graph_gte_pretrained", "gammagl/models/graph_gte_pretrained.py")
    graph_gte = _load("graph_gte_pretrained_test_model", "gammagl/models/graph_gte.py")
    return graph_gte.GraphGTE, pretrained


def _tiny_checkpoint(model):
    layer = model.encoder_layers[0]
    values = {
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
    return {key: torch.arange(value.numel(), dtype=torch.float32).reshape(value.shape)
            if key.startswith("new.encoder") else torch.ones_like(value.detach())
            for key, value in values.items()}


def test_sha256_file_is_content_hash_and_detects_mismatch(tmp_path):
    _, pretrained = _modules()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"strict-checksum")
    assert pretrained.sha256_file(checkpoint) == hashlib.sha256(b"strict-checksum").hexdigest()
    assert pretrained.sha256_file(checkpoint) != pretrained.GTE_CHECKPOINT_SHA256


def test_download_rejects_checkpoint_sha256_mismatch(tmp_path, monkeypatch):
    _, pretrained = _modules()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"tampered")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(checkpoint))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        pretrained.download_official_gte_checkpoint()


def test_explicit_converter_has_full_coverage_and_parameter_equality(tmp_path):
    GraphGTE, pretrained = _modules()
    source_model = GraphGTE(vocab_size=7, output_dim=2, hidden_size=16,
                            num_hidden_layers=1, num_attention_heads=4,
                            intermediate_size=32, max_position_embeddings=16)
    tensors = _tiny_checkpoint(source_model)
    checkpoint = tmp_path / "tiny.safetensors"
    safetensors_torch.save_file(tensors, str(checkpoint))
    model = GraphGTE(vocab_size=7, output_dim=2, hidden_size=16,
                     num_hidden_layers=1, num_attention_heads=4,
                     intermediate_size=32, max_position_embeddings=16)
    report = pretrained.load_pretrained_encoder(model, checkpoint)
    assert report["coverage"] == 1.0
    assert report["missing"] == []
    assert report["shape_mismatches"] == []
    assert report["unexpected"] == []
    assert "text-vocabulary embedding" in report["ignored"]["new.embeddings.word_embeddings.weight"]
    assert not np.array_equal(
        tlx.convert_to_numpy(model.token_embeddings.embeddings),
        tensors["new.embeddings.word_embeddings.weight"].numpy())
    layer = model.encoder_layers[0]
    checks = {
        "new.encoder.layer.0.attention.qkv_proj.weight": layer.attention.qkv_proj.weights,
        "new.encoder.layer.0.attention.o_proj.weight": layer.attention.out_proj.weights,
        "new.encoder.layer.0.mlp.up_gate_proj.weight": layer.mlp.up_gate_proj.weights,
        "new.encoder.layer.0.mlp_ln.weight": layer.mlp_norm.gamma,
    }
    for key, target in checks.items():
        np.testing.assert_array_equal(tlx.convert_to_numpy(target), tensors[key].numpy())


def test_converter_rejects_shape_mismatch(tmp_path):
    GraphGTE, pretrained = _modules()
    source = GraphGTE(vocab_size=7, output_dim=2, hidden_size=16,
                      num_hidden_layers=1, num_attention_heads=4,
                      intermediate_size=32, max_position_embeddings=16)
    tensors = _tiny_checkpoint(source)
    tensors["new.encoder.layer.0.attention.qkv_proj.weight"] = torch.zeros((1, 1))
    checkpoint = tmp_path / "bad.safetensors"
    safetensors_torch.save_file(tensors, str(checkpoint))
    with pytest.raises(RuntimeError, match="Incomplete GTE encoder conversion"):
        pretrained.load_pretrained_encoder(source, checkpoint)


def test_official_checkpoint_converter_and_hf_encoder_equivalence():
    """Real pinned GTE checkpoint check; opt in with GTE_CHECKPOINT_PATH."""
    checkpoint_path = os.environ.get("GTE_CHECKPOINT_PATH")
    if not checkpoint_path:
        pytest.skip("set GTE_CHECKPOINT_PATH to run the pinned-GTE integration test")
    transformers = pytest.importorskip("transformers")
    GraphGTE, pretrained = _modules()
    assert pretrained.sha256_file(checkpoint_path) == pretrained.GTE_CHECKPOINT_SHA256
    model = GraphGTE(vocab_size=19, output_dim=2)
    report = pretrained.load_pretrained_encoder(model, checkpoint_path)
    assert report["coverage"] == 1.0
    assert report["missing"] == []
    assert report["shape_mismatches"] == []

    from safetensors import safe_open
    with safe_open(checkpoint_path, framework="pt", device="cpu") as source:
        equality_checks = {
            "new.encoder.layer.0.attention.qkv_proj.weight":
                model.encoder_layers[0].attention.qkv_proj.weights,
            "new.encoder.layer.0.attention.o_proj.weight":
                model.encoder_layers[0].attention.out_proj.weights,
            "new.encoder.layer.6.mlp.up_gate_proj.weight":
                model.encoder_layers[6].mlp.up_gate_proj.weights,
            "new.encoder.layer.11.mlp_ln.weight": model.encoder_layers[11].mlp_norm.gamma,
        }
        for key, target in equality_checks.items():
            np.testing.assert_array_equal(
                source.get_tensor(key).numpy(), tlx.convert_to_numpy(target))

    reference = transformers.AutoModel.from_pretrained(
        pretrained.GTE_MODEL_ID, revision=pretrained.GTE_REVISION,
        trust_remote_code=True, local_files_only=True).eval()
    model.set_eval()
    torch.manual_seed(11)
    inputs = torch.randn(1, 5, 768)
    mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)
    with torch.no_grad():
        reference_outputs = reference(
            inputs_embeds=inputs, attention_mask=mask,
            output_hidden_states=True, return_dict=True).hidden_states
        _, tlx_outputs = model.encode_embeddings(
            inputs, mask, return_hidden_states=True)
    errors = [float(np.max(np.abs(
        reference_output.detach().numpy() - tlx.convert_to_numpy(tlx_output))))
              for reference_output, tlx_output in zip(reference_outputs, tlx_outputs)]
    assert len(errors) == 13
    assert errors[1] < 5e-5  # layer 0
    assert errors[7] < 5e-5  # middle layer (hidden_states includes embeddings)
    assert errors[-1] < 5e-5  # final layer


def test_official_from_pretrained_smoke_updates_encoder():
    if not os.environ.get("GTE_CHECKPOINT_PATH"):
        pytest.skip("set GTE_CHECKPOINT_PATH to run the pinned-GTE integration test")
    GraphGTE, _ = _modules()
    model = GraphGTE.from_pretrained(vocab_size=19, output_dim=2)
    manifest = model.manifest()
    assert type(model).__name__ == "GraphGTE"
    assert manifest["pretrained"] is True
    assert manifest["encoder_coverage"] == 1.0
    parameter = model.encoder_layers[0].attention.qkv_proj.weights
    before = parameter.detach().clone()
    optimizer = torch.optim.SGD(model.trainable_weights, lr=1e-4)
    output = model(
        torch.tensor([[2, 3, 4, 1]], dtype=torch.long),
        torch.tensor([[1, 1, 1, 0]], dtype=torch.long), task="supervised")
    loss = (output * output).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert parameter.grad is not None
    assert not torch.equal(before, parameter.detach())
