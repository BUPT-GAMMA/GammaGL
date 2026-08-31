"""Strict official GTE checkpoint loading for the native TLX GraphGTE."""
import hashlib
import json
from pathlib import Path


GTE_MODEL_ID = "Alibaba-NLP/gte-multilingual-base"
GTE_REVISION = "9bbca17d9273fd0d03d5725c7a4b0f6b45142062"
GTE_CHECKPOINT = "model.safetensors"
GTE_CHECKPOINT_SHA256 = "f5a35a10faa54da7717870af1517c9b41e9bd8e3880bc5a8e9363d4c3c63e9b0"


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_official_gte_checkpoint(cache_dir=None):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(GTE_MODEL_ID, GTE_CHECKPOINT, revision=GTE_REVISION,
                           cache_dir=cache_dir)
    actual = sha256_file(path)
    if actual != GTE_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"GTE checkpoint SHA-256 mismatch: expected {GTE_CHECKPOINT_SHA256}, got {actual}.")
    return Path(path), actual


def load_official_gte_config(cache_dir=None):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(GTE_MODEL_ID, "config.json", revision=GTE_REVISION,
                           cache_dir=cache_dir)
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"hidden_size": 768, "num_hidden_layers": 12,
                "num_attention_heads": 12, "intermediate_size": 3072,
                "pack_qkv": True, "position_embedding_type": "rope",
                "rope_theta": 20000, "max_position_embeddings": 8192,
                "hidden_act": "gelu", "layer_norm_type": "layer_norm",
                "attention_probs_dropout_prob": 0.0,
                "use_memory_efficient_attention": False}
    for key, value in required.items():
        if config.get(key) != value:
            raise RuntimeError(f"Official GTE config mismatch for {key}: {config.get(key)!r}.")
    if config.get("rope_scaling") != {"factor": 8.0, "type": "ntk"}:
        raise RuntimeError("Official GTE RoPE scaling config mismatch.")
    return config


def _assign_tensor(target, value, key, transpose=False):
    """Assign one semantically mapped PyTorch tensor to its TLX variable."""
    import tensorlayerx as tlx

    value = value.detach().cpu().numpy()
    if transpose:
        if value.ndim != 2:
            raise ValueError(f"Only matrix mappings may transpose: {key}.")
        value = value.T
    if tuple(value.shape) != tuple(target.shape):
        raise ValueError(
            f"{key}: mapped shape {tuple(value.shape)} does not match "
            f"TLX target {tuple(target.shape)}.")
    value = tlx.convert_to_tensor(value, dtype=target.dtype)
    if hasattr(target, "assign"):
        target.assign(value)
    else:
        target.data.copy_(value)


def load_pretrained_encoder(model, checkpoint_path):
    """Load the explicitly mapped official encoder tensors into ``GraphGTE``."""
    from safetensors import safe_open

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        source = {key: handle.get_tensor(key) for key in handle.keys()}
    ignored = {
        "new.embeddings.word_embeddings.weight":
            "text-vocabulary embedding is incompatible with Graph BPE tokens",
        "classifier.weight": "official classification head is not a GraphTokenizer task head",
        "classifier.bias": "official classification head is not a GraphTokenizer task head",
    }
    # Every entry is a source-key-to-TLX-variable semantic mapping.  The
    # official NewModel and TLX torch backend both store Linear kernels as
    # [out_features, in_features], hence transpose=False is deliberate rather
    # than inferred from shape.
    mapping = {
        "new.embeddings.token_type_embeddings.weight": (model.token_type_embeddings.embeddings, False, "embedding"),
        "new.embeddings.LayerNorm.weight": (model.embedding_norm.gamma, False, "layer_norm_gamma"),
        "new.embeddings.LayerNorm.bias": (model.embedding_norm.beta, False, "layer_norm_beta"),
    }
    for index, layer in enumerate(model.encoder_layers):
        prefix = f"new.encoder.layer.{index}"
        mapping.update({
            f"{prefix}.attention.qkv_proj.weight": (layer.attention.qkv_proj.weights, False, "packed_qkv_QKV"),
            f"{prefix}.attention.qkv_proj.bias": (layer.attention.qkv_proj.biases, False, "packed_qkv_bias_QKV"),
            f"{prefix}.attention.o_proj.weight": (layer.attention.out_proj.weights, False, "attention_output"),
            f"{prefix}.attention.o_proj.bias": (layer.attention.out_proj.biases, False, "attention_output_bias"),
            f"{prefix}.attn_ln.weight": (layer.attention_norm.gamma, False, "post_attention_layer_norm_gamma"),
            f"{prefix}.attn_ln.bias": (layer.attention_norm.beta, False, "post_attention_layer_norm_beta"),
            f"{prefix}.mlp.up_gate_proj.weight": (layer.mlp.up_gate_proj.weights, False, "packed_up_gate_up_then_gate"),
            f"{prefix}.mlp.down_proj.weight": (layer.mlp.down_proj.weights, False, "mlp_down"),
            f"{prefix}.mlp.down_proj.bias": (layer.mlp.down_proj.biases, False, "mlp_down_bias"),
            f"{prefix}.mlp_ln.weight": (layer.mlp_norm.gamma, False, "post_mlp_layer_norm_gamma"),
            f"{prefix}.mlp_ln.bias": (layer.mlp_norm.beta, False, "post_mlp_layer_norm_beta"),
        })
    expected = set(mapping)
    missing = sorted(expected - set(source))
    unexpected = sorted(set(source) - expected - set(ignored))
    mismatches = []
    loaded = []
    for key, (target, transpose, _semantic) in mapping.items():
        if key not in source:
            continue
        try:
            _assign_tensor(target, source[key], key, transpose=transpose)
        except ValueError:
            mismatches.append({"key": key, "source": tuple(source[key].shape),
                               "target": tuple(target.shape), "transpose": transpose})
            continue
        loaded.append(key)
    report = {"expected": sorted(expected), "loaded": sorted(loaded), "missing": missing,
              "unexpected": unexpected, "ignored": ignored, "shape_mismatches": mismatches,
              "coverage": len(loaded) / len(expected) if expected else 1.0}
    if missing or unexpected or mismatches or report["coverage"] != 1.0:
        raise RuntimeError(f"Incomplete GTE encoder conversion: {report}")
    return report
