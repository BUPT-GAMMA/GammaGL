"""Torch-only training protocol aligned with the GraphTokenizer paper."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


PAPER_RUNTIME_REQUIREMENTS = {
    "torch": "2.1.2",
    "cuda": "12.1",
    "dgl": "2.4.0",
    "torch_geometric": "2.4.0",
}

PAPER_SOURCE_REVISION = "98343a6b025a48fbb6859cd812a12b81ec3ac3cc"

PAPER_ARCHITECTURES = {
    "bert": {
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "max_position_embeddings": 8096,
        "parameter_estimate": 16_000_000,
    },
    "gte": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "max_position_embeddings": 8192,
        "parameter_estimate": 115_000_000,
    },
}


def _estimate_training_peak_bytes(
        encoder_type: str, micro_batch_size: int,
        sequence_length: int, activation_bytes_per_element: int = 4) -> int:
    """Conservative optimizer/parameter and saved-activation estimate."""
    architecture = PAPER_ARCHITECTURES[str(encoder_type).lower()]
    batch_size = int(micro_batch_size)
    sequence_length = int(sequence_length)
    layers = int(architecture["num_hidden_layers"])
    heads = int(architecture["num_attention_heads"])
    hidden_size = int(architecture["hidden_size"])
    intermediate_size = int(architecture["intermediate_size"])
    activation_bytes_per_element = int(activation_bytes_per_element)
    if activation_bytes_per_element not in {2, 3, 4}:
        raise ValueError(
            "activation_bytes_per_element must be 2, 3, or 4.")
    # FP32 parameters, gradients, and two AdamW moments.
    parameter_bytes = int(architecture["parameter_estimate"]) * 16
    # Autograd retains attention probabilities and related score tensors.
    attention_bytes = (
        2 * batch_size * heads * sequence_length * sequence_length
        * layers * activation_bytes_per_element
    )
    # QKV/residual/normalization/FFN intermediates, including gated GTE FFN.
    activation_width = 8 * hidden_size + 3 * intermediate_size
    activation_bytes = (
        batch_size * sequence_length * layers * activation_width
        * activation_bytes_per_element
    )
    return int(parameter_bytes + attention_bytes + activation_bytes)


def _resolve_memory_plan(
        encoder_type: str,
        effective_batch_size: int,
        sequence_length: int,
        available_bytes: int,
        safety_fraction: float = 0.8,
        activation_bytes_per_element: int = 4) -> Dict[str, Any]:
    """Choose a divisor micro-batch while preserving the paper batch size."""
    encoder_type = str(encoder_type).lower()
    if encoder_type not in PAPER_ARCHITECTURES:
        raise ValueError("encoder_type must be 'bert' or 'gte'.")
    effective_batch_size = int(effective_batch_size)
    sequence_length = int(sequence_length)
    available_bytes = int(available_bytes)
    if effective_batch_size <= 0 or sequence_length <= 0:
        raise ValueError("Batch size and sequence length must be positive.")
    if available_bytes <= 0:
        raise ValueError("available_bytes must be positive.")
    if not 0 < float(safety_fraction) <= 1:
        raise ValueError("safety_fraction must be in (0, 1].")
    budget_bytes = int(available_bytes * float(safety_fraction))
    candidates = [
        value for value in range(effective_batch_size, 0, -1)
        if effective_batch_size % value == 0
    ]
    for micro_batch_size in candidates:
        estimated = _estimate_training_peak_bytes(
            encoder_type, micro_batch_size, sequence_length,
            activation_bytes_per_element=activation_bytes_per_element)
        if estimated <= budget_bytes:
            return {
                "effective_batch_size": effective_batch_size,
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": (
                    effective_batch_size // micro_batch_size),
                "sequence_length": sequence_length,
                "available_bytes": available_bytes,
                "budget_bytes": budget_bytes,
                "estimated_peak_bytes": estimated,
                "safety_fraction": float(safety_fraction),
                "activation_bytes_per_element": int(
                    activation_bytes_per_element),
            }
    minimum = _estimate_training_peak_bytes(
        encoder_type, 1, sequence_length,
        activation_bytes_per_element=activation_bytes_per_element)
    raise RuntimeError(
        "Estimated dense attention training memory is unsafe even with "
        f"micro_batch_size=1: estimated={minimum / 1024 ** 3:.2f} GiB, "
        f"budget={budget_bytes / 1024 ** 3:.2f} GiB, "
        f"encoder={encoder_type}, sequence_length={sequence_length}.")


def _load_paper_model_classes():
    from gammagl.models.graph_bert import GraphBERT
    from gammagl.models.graph_gte import GraphGTE

    return GraphBERT, GraphGTE


def require_paper_runtime(require_cuda: bool = True):
    if os.environ.get("TL_BACKEND", "torch").lower() != "torch":
        raise RuntimeError("Paper protocol requires TL_BACKEND=torch.")
    import tensorlayerx as tlx
    if str(tlx.BACKEND).lower() != "torch":
        raise RuntimeError(
            "Paper protocol requires the TensorLayerX PyTorch backend; "
            f"received {tlx.BACKEND!r}.")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Paper protocol requires PyTorch.") from error
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("Paper protocol requires a CUDA-enabled PyTorch runtime.")
    return torch


def _create_paper_model(
        encoder_type: str,
        vocab_size: int,
        pad_token_id: int,
        task_type: str,
        output_dim: int,
        pooling: str = "mean",
        model_config: Dict[str, Any] | None = None,
        strict_architecture: bool = True,
        allow_random_gte_init: bool = False,
        pretrained_cache_dir: str | None = None):
    GraphBERT, GraphGTE = _load_paper_model_classes()
    encoder_type = str(encoder_type).lower()
    if encoder_type not in {"bert", "gte"}:
        raise ValueError("encoder_type must be 'bert' or 'gte'.")
    model_class = GraphBERT if encoder_type == "bert" else GraphGTE
    kwargs = {
        "vocab_size": int(vocab_size),
        "output_dim": int(output_dim),
        "pad_token_id": int(pad_token_id),
        "task_type": task_type,
        "pooling": pooling,
    }
    if not strict_architecture:
        allowed = {
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "max_position_embeddings",
            "dropout_rate",
            "attention_dropout_rate",
            "layer_norm_eps",
            "task_dropout",
            "rope_theta",
            "rope_scaling_factor",
        }
        kwargs.update({
            key: value
            for key, value in dict(model_config or {}).items()
            if key in allowed
        })
    if encoder_type == "gte" and not allow_random_gte_init:
        # A paper GTE run is a reproduction only with the pinned official
        # encoder.  Any download/config/conversion error intentionally escapes.
        return GraphGTE.from_pretrained(cache_dir=pretrained_cache_dir, **kwargs)
    if encoder_type == "bert" and strict_architecture:
        kwargs["max_position_embeddings"] = PAPER_ARCHITECTURES["bert"][
            "max_position_embeddings"]
    model = model_class(**kwargs)
    if encoder_type == "gte":
        model.pretrained_manifest = {
            "pretrained": False,
            "reproduction": False,
            "graph_embedding_init": "new_truncated_normal(stddev=0.02)",
            "task_head_init": "new_truncated_normal(stddev=0.02)",
        }
    return model


def _require_fp32_parameters(torch, model) -> None:
    non_fp32 = [
        name
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != torch.float32
    ]
    if non_fp32:
        preview = ", ".join(non_fp32[:3])
        raise RuntimeError(
            "Paper protocol requires FP32 model parameters before AdamW; "
            f"received non-FP32 parameters: {preview}.")


def _torch():
    return require_paper_runtime(require_cuda=True)


def validate_paper_runtime_versions(versions: Dict[str, Any]) -> None:
    for field, expected in PAPER_RUNTIME_REQUIREMENTS.items():
        actual = str(versions.get(field, "<missing>"))
        comparable = actual.split("+", 1)[0] if field in {"torch", "dgl"} else actual
        if comparable != expected:
            raise RuntimeError(
                f"Paper runtime requires {field}=={expected}; received {actual}.")


def paper_runtime_status() -> Dict[str, Any]:
    torch = _torch()
    try:
        import dgl
        import numpy as np
        import ogb
        import torch_geometric
    except ImportError as error:
        raise RuntimeError(
            "Paper protocol requires DGL, PyTorch Geometric, numpy and OGB.") from error
    device_index = torch.cuda.current_device()
    status = {
        "status": "ok",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dgl": dgl.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_device": torch.cuda.get_device_name(device_index),
        "numpy": np.__version__,
        "ogb": ogb.__version__,
    }
    validate_paper_runtime_versions(status)
    status["requirements"] = dict(PAPER_RUNTIME_REQUIREMENTS)
    return status


def validate_paper_args(args, spec) -> None:
    if spec.canonical_name not in {"qm9", "molhiv", "peptides-struct"}:
        raise ValueError(
            "Paper protocol supports only QM9, OGBG-molhiv and Peptides-struct.")
    if spec.canonical_name == "qm9":
        target_property = getattr(args, "target_property", None)
        if target_property not in {None, "homo"}:
            raise ValueError(
                "QM9 strict paper protocol evaluates the single HOMO target; "
                "use --target-property=homo or omit the option.")
    requested_model = getattr(args, "model", None)
    if requested_model and requested_model not in {"bert", "gte"}:
        raise ValueError("Paper protocol model must be 'bert' or 'gte'.")
    if requested_model:
        expected_position_limit = PAPER_ARCHITECTURES.get(
            requested_model, {}).get("max_position_embeddings")
        if expected_position_limit is not None and int(getattr(
                args, "max_position_embeddings", expected_position_limit)) != expected_position_limit:
            raise ValueError(
                "Paper protocol requires "
                f"--max-position-embeddings={expected_position_limit} for "
                f"{requested_model}.")
    if requested_model == "gte" and bool(getattr(
            args, "allow_random_gte_init", False)):
        raise ValueError(
            "Strict paper protocol requires the pinned pretrained GTE weights.")


def paper_run_seeds(args) -> List[int]:
    runs = getattr(args, "runs", None)
    runs = 5 if runs is None else int(runs)
    if runs != 5:
        raise ValueError("Paper protocol requires exactly five independent runs.")
    start_seed = 42 if getattr(args, "seed", None) is None else int(args.seed)
    return [start_seed + index for index in range(runs)]


def select_best_epoch(history: Sequence[Dict[str, Any]], higher_is_better: bool) -> int:
    if not history:
        raise ValueError("Cannot select a checkpoint from an empty training history.")
    key = lambda entry: float(entry["val_metric"])
    return int((max if higher_is_better else min)(history, key=key)["epoch"])


def _should_skip_finetuning(
        resume_phase, stale_epochs: int, patience: int) -> bool:
    return resume_phase == "finetune_complete" or (
        resume_phase == "finetune"
        and int(stale_epochs) >= int(patience)
    )


def _paper_token_augmentation() -> Dict[str, Any]:
    return {
        "pretrain": {
            "random_swap": {
                "probability": 0.5, "ratio": 0.1, "window_size": 3}},
        "finetune": {
            "random_swap": {
                "probability": 0.4, "ratio": 0.05, "window_size": 3},
            "sequence_masking": {"probability": 0.3, "ratio": 0.05},
        },
        "evaluation": "none",
    }


def paper_reference_options(spec, encoder: str) -> Dict[str, Any]:
    """Return the pinned official settings used for strict comparison."""
    encoder = str(encoder).lower()
    if encoder not in PAPER_ARCHITECTURES:
        raise ValueError("Paper protocol encoder must be 'bert' or 'gte'.")
    dataset = spec.canonical_name
    pretrain_lr = (
        1e-4 if encoder == "bert" or dataset == "peptides-struct" else 5e-5)
    return {
        "source_revision": PAPER_SOURCE_REVISION,
        "encoder": encoder,
        "serialization": "feuler",
        "pretrain_epochs": 200,
        "finetune_epochs": 200,
        "pretrain_lr": pretrain_lr,
        "finetune_lr": 5e-5 if dataset == "molhiv" else 1e-5,
        "batch_size": 16 if dataset == "peptides-struct" else 32,
        "weight_decay": 0.1,
        "pretrain_warmup_ratio": 0.12,
        "finetune_warmup_ratio": 0.025,
        "pretrain_max_grad_norm": 2.0,
        "finetune_max_grad_norm": 0.5,
        "mask_prob": 0.09,
        "patience": 20,
        "max_position_embeddings": int(
            PAPER_ARCHITECTURES[encoder]["max_position_embeddings"]),
        "max_sequence_length": 768 if encoder == "bert" else 8096,
        "num_merges": 2000,
        "min_frequency": 2,
        "num_realizations": 100,
        "aggregation_mode": "avg",
        "pooling": "mean",
        "amp_dtype": "off",
        "allow_tf32": False,
        "training_loss": "default",
        "molhiv_pos_weight": False,
        "token_augmentation": _paper_token_augmentation(),
        "gaussian_noise": {"probability": 0.3, "std": 0.01},
        "pretraining_graph_corpus": (
            "peptides-func" if dataset == "peptides-struct" else dataset),
    }


def validate_paper_training_options(options, spec) -> None:
    expected = paper_reference_options(spec, options["encoder"])
    for name, expected_value in expected.items():
        actual = options.get(name)
        if actual != expected_value:
            raise ValueError(
                f"Strict paper protocol requires {name}={expected_value!r}; "
                f"received {actual!r}.")


def paper_training_options(args, spec=None) -> Dict[str, Any]:
    encoder = str(args.model).lower()
    options = {
        "encoder": encoder,
        "serialization": args.serialization,
        "pretrain_epochs": args.pretrain_epoch,
        "finetune_epochs": args.n_epoch,
        "pretrain_lr": args.pretrain_lr,
        "finetune_lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "pretrain_warmup_ratio": args.pretrain_warmup_ratio,
        "finetune_warmup_ratio": args.finetune_warmup_ratio,
        "pretrain_max_grad_norm": args.pretrain_max_grad_norm,
        "finetune_max_grad_norm": args.finetune_max_grad_norm,
        "mask_prob": args.mask_prob,
        "patience": args.patience,
        "max_position_embeddings": args.max_position_embeddings,
        "max_sequence_length": 768 if encoder == "bert" else 8096,
        "num_merges": int(getattr(args, "num_merges", 2000)),
        "min_frequency": int(getattr(args, "min_frequency", 2)),
        "num_realizations": int(getattr(args, "num_realizations", 100)),
        "aggregation_mode": str(getattr(args, "aggregation_mode", "avg")),
        "pooling": str(getattr(args, "pooling", "mean")),
        "amp_dtype": "off",
        "allow_tf32": False,
        "training_loss": "default",
        "molhiv_pos_weight": False,
    }
    cli_overrides = {
        "amp_dtype": getattr(args, "paper_amp", None),
        "allow_tf32": getattr(args, "paper_tf32", None),
        "training_loss": getattr(args, "paper_loss", None),
        "molhiv_pos_weight": getattr(args, "molhiv_pos_weight", None),
    }
    options.update({
        key: value for key, value in cli_overrides.items()
        if value is not None
    })
    if spec is not None:
        options.update({
            "source_revision": PAPER_SOURCE_REVISION,
            "token_augmentation": _paper_token_augmentation(),
            "gaussian_noise": {"probability": 0.3, "std": 0.01},
            "pretraining_graph_corpus": (
                "peptides-func" if spec.canonical_name == "peptides-struct"
                else spec.canonical_name),
        })
    return options


def set_paper_seed(seed: int, torch) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _pad_sequences(sequences: Sequence[Sequence[int]], max_length: int, pad_token_id: int):
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    if any(len(sequence) > max_length for sequence in sequences):
        raise ValueError(
            f"A token sequence exceeds the configured max_length={max_length}.")
    input_ids = [list(map(int, sequence)) + [pad_token_id] * (max_length - len(sequence))
                 for sequence in sequences]
    masks = [[1] * len(sequence) + [0] * (max_length - len(sequence))
             for sequence in sequences]
    return input_ids, masks


def _encode_splits(
        tokenizer, splits, max_position_embeddings: int,
        encoding_batch_size: int = 2048,
        max_sequence_length: int | None = None,
        num_realizations: int = 1):
    max_sequence_length = int(
        max_position_embeddings if max_sequence_length is None
        else max_sequence_length)
    if max_sequence_length <= 0 or max_sequence_length > max_position_embeddings:
        raise ValueError(
            "max_sequence_length must be positive and no greater than "
            "max_position_embeddings.")
    num_realizations = int(num_realizations)
    if num_realizations <= 0:
        raise ValueError("num_realizations must be positive.")

    def truncate(sequence):
        sequence = list(map(int, sequence))
        if len(sequence) <= max_sequence_length:
            return sequence
        return [
            *sequence[:max_sequence_length - 1],
            int(tokenizer.special_tokens.sep_token_id),
        ]

    def start_nodes(graph):
        method = getattr(tokenizer, "realization_start_nodes", None)
        if callable(method):
            return method(graph, num_realizations)
        if num_realizations == 1:
            return [None]
        value = getattr(graph, "num_nodes", None)
        if callable(value):
            value = value()
        total_nodes = int(value)
        actual_samples = min(num_realizations, total_nodes)
        step = max(1, total_nodes // actual_samples)
        return [(index * step) % total_nodes for index in range(actual_samples)]

    encoded = {}
    global_max_length = 1
    for split_name, graphs in splits.items():
        if not graphs:
            raise ValueError(
                f"Paper protocol requires a non-empty {split_name} split.")
        results = []
        graph_ids = []
        if num_realizations == 1 and hasattr(tokenizer, "batch_encode_graphs"):
            encoding_batch_size = max(1, int(encoding_batch_size))
            for start in range(0, len(graphs), encoding_batch_size):
                graph_batch = graphs[start:start + encoding_batch_size]
                batch_results = tokenizer.batch_encode_graphs(graph_batch)
                if len(batch_results) != len(graph_batch):
                    raise RuntimeError(
                        "Tokenizer batch encoding returned the wrong count.")
                results.extend(batch_results)
                graph_ids.extend(range(start, start + len(graph_batch)))
        else:
            for graph_id, graph in enumerate(graphs):
                for start_node in start_nodes(graph):
                    result = (
                        tokenizer.encode_graph(graph)
                        if start_node is None
                        else tokenizer.encode_graph(graph, start_node=start_node))
                    results.append(result)
                    graph_ids.append(graph_id)
        sequences = [truncate(result.input_ids) for result in results]
        maximum = max((len(sequence) for sequence in sequences), default=1)
        global_max_length = max(global_max_length, maximum)
        encoded[split_name] = {
            "input_ids": sequences,
            "serialized_token_ids": [
                list(map(int, getattr(result, "serialized_token_ids", [])))
                for result in results
            ],
            "labels": [list(graphs[graph_id].y) for graph_id in graph_ids],
            "graph_ids": graph_ids,
        }
    return encoded, global_max_length


def _load_or_encode_splits(
        tokenizer, splits, max_position_embeddings: int,
        cache_path, cache_key: str, max_sequence_length: int | None = None,
        num_realizations: int = 1):
    max_sequence_length = int(
        max_position_embeddings if max_sequence_length is None
        else max_sequence_length)
    cache_path = Path(cache_path)
    if cache_path.is_file():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if (
                    cached.get("cache_version") == 2
                    and cached.get("cache_key") == str(cache_key)
                    and cached.get("max_position_embeddings")
                    == int(max_position_embeddings)
                    and cached.get("max_sequence_length")
                    == max_sequence_length
                    and cached.get("num_realizations")
                    == int(num_realizations)):
                return cached["encoded"], int(cached["global_max_length"])
        except (OSError, EOFError, AttributeError, KeyError, pickle.UnpicklingError):
            pass
    encoded, global_max_length = _encode_splits(
        tokenizer, splits, int(max_position_embeddings),
        max_sequence_length=max_sequence_length,
        num_realizations=num_realizations)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(
        f".{cache_path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump({
            "cache_version": 2,
            "cache_key": str(cache_key),
            "max_position_embeddings": int(max_position_embeddings),
            "max_sequence_length": max_sequence_length,
            "num_realizations": int(num_realizations),
            "global_max_length": int(global_max_length),
            "encoded": encoded,
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, cache_path)
    return encoded, global_max_length


def _encoded_splits_cache_key(
        tokenizer, splits, max_position_embeddings: int,
        max_sequence_length: int | None = None,
        num_realizations: int = 1) -> str:
    split_digests = {}
    for split_name, graphs in splits.items():
        digest = hashlib.sha256()
        for graph in graphs:
            fields = []
            for field_name in (
                    "edge_index", "x", "edge_attr", "y", "num_nodes",
                    "tokens"):
                value = getattr(graph, field_name, None)
                if hasattr(value, "detach"):
                    value = value.detach()
                if hasattr(value, "cpu"):
                    value = value.cpu()
                if hasattr(value, "tolist"):
                    value = value.tolist()
                fields.append((field_name, value))
            digest.update(pickle.dumps(fields, protocol=4))
        split_digests[split_name] = digest.hexdigest()
    state = {
        "cache_version": 2,
        "tokenizer_cache_key": getattr(tokenizer, "_cache_key", None),
        "serializer": getattr(tokenizer.serializer, "name", None),
        "frequency_map": sorted(tokenizer.serializer.frequency_map.items()),
        "merge_rules": list(tokenizer.bpe.codebook.merge_rules),
        "vocab_size": int(tokenizer.bpe.codebook.vocab_size),
        "split_sizes": {
            split_name: len(graphs) for split_name, graphs in splits.items()
        },
        "split_digests": split_digests,
        "max_position_embeddings": int(max_position_embeddings),
        "max_sequence_length": int(
            max_position_embeddings if max_sequence_length is None
            else max_sequence_length),
        "num_realizations": int(num_realizations),
    }
    return hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest()


def _prepare_labels(encoded, spec, target_property):
    if spec.canonical_name == "qm9":
        source_width = int(spec.output_dim)
        source_properties = list(spec.label_keys)
        if len(source_properties) != source_width:
            raise ValueError(
                f"QM9 requires exactly {source_width} target property names; "
                f"received {len(source_properties)}.")
        selected_target = "homo" if target_property is None else str(target_property)
        if selected_target != "homo" or selected_target not in source_properties:
            raise ValueError("QM9 paper evaluation requires the HOMO target.")
        target_index = source_properties.index(selected_target)
        if not encoded["train"]["labels"]:
            raise ValueError("QM9 training split cannot be empty.")
        for split_name, split in encoded.items():
            for row_index, row in enumerate(split["labels"]):
                if len(row) != source_width:
                    raise ValueError(
                        f"QM9 {split_name} label row {row_index} must contain "
                        f"exactly {source_width} targets; received {len(row)}.")
                if not math.isfinite(float(row[target_index])):
                    raise ValueError(
                        f"QM9 {split_name} label row {row_index} contains a "
                        "non-finite target.")
        train_values = [
            float(row[target_index]) for row in encoded["train"]["labels"]]
        means = [sum(train_values) / len(train_values)]
        variance = sum(
            (value - means[0]) ** 2 for value in train_values
        ) / len(train_values)
        stds = [max(math.sqrt(variance), 1e-12)]
        for split in encoded.values():
            split["labels"] = [
                [(float(row[target_index]) - means[0]) / stds[0]]
                for row in split["labels"]
            ]
        return {
            "mean": means,
            "std": stds,
            "target_properties": [selected_target],
        }, 1
    if spec.canonical_name == "peptides-struct":
        output_dim = int(spec.output_dim)
        target_properties = list(getattr(spec, "label_keys", ()))
        if len(target_properties) != output_dim:
            target_properties = [f"target_{index}" for index in range(output_dim)]
        for split_name, split in encoded.items():
            for row_index, row in enumerate(split["labels"]):
                if len(row) != output_dim:
                    raise ValueError(
                        f"Peptides-struct {split_name} label row {row_index} "
                        f"must contain exactly {output_dim} targets; received "
                        f"{len(row)}.")
        train_rows = [list(map(float, row)) for row in encoded["train"]["labels"]]
        means, stds = [], []
        for index in range(output_dim):
            values = [row[index] for row in train_rows if math.isfinite(row[index])]
            if not values:
                raise ValueError(
                    f"Peptides-struct training target {index} has no finite labels.")
            target_mean = sum(values) / len(values)
            variance = sum(
                (value - target_mean) ** 2 for value in values) / len(values)
            means.append(target_mean)
            stds.append(max(math.sqrt(variance), 1e-12))
        for split in encoded.values():
            split["labels"] = [
                [
                    ((float(value) - means[index]) / stds[index]
                     if math.isfinite(float(value)) else float("nan"))
                    for index, value in enumerate(row)
                ]
                for row in split["labels"]
            ]
        return {
            "mean": means,
            "std": stds,
            "target_properties": target_properties,
        }, output_dim
    if spec.canonical_name == "molhiv":
        return None, 2
    return None, spec.output_dim


def _denormalize_labels(torch, values, normalizer):
    means = torch.as_tensor(
        normalizer["mean"], dtype=values.dtype, device=values.device)
    stds = torch.as_tensor(
        normalizer["std"], dtype=values.dtype, device=values.device)
    if values.ndim < 1 or values.shape[-1] != len(means):
        raise ValueError(
            "Normalized label width does not match the target normalizer.")
    return values * stds + means


def _metric_semantics(spec, normalizer):
    """Describe the training-loss and reported-metric spaces explicitly."""
    if spec.canonical_name in {"qm9", "peptides-struct"}:
        normalization = {"type": "train_split_zscore"}
        if normalizer is not None:
            normalization.update({
                "mean": list(normalizer["mean"]),
                "std": list(normalizer["std"]),
            })
        return {
            "loss_space": "standardized",
            "metric_space": "raw",
            "target_normalization": normalization,
        }
    if spec.canonical_name == "molhiv":
        return {
            "loss_space": "logits",
            "metric_space": "probability",
            "target_normalization": {"type": "none"},
        }
    return {
        "loss_space": "logits",
        "metric_space": "probability",
        "target_normalization": {"type": "none"},
    }


class _LengthBucketBatchSampler:
    """Shuffle examples, then form locally length-sorted dynamic batches."""

    def __init__(
            self, torch, lengths, batch_size: int, shuffle: bool,
            pool_size_multiplier: int = 50):
        self.torch = torch
        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.pool_size = max(
            self.batch_size, self.batch_size * int(pool_size_multiplier))

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            indices = self.torch.randperm(len(self.lengths)).tolist()
        else:
            indices = list(range(len(self.lengths)))
        batches = []
        for start in range(0, len(indices), self.pool_size):
            pool = indices[start:start + self.pool_size]
            if self.shuffle:
                pool.sort(key=lambda index: self.lengths[index], reverse=True)
            for offset in range(0, len(pool), self.batch_size):
                batches.append(pool[offset:offset + self.batch_size])
        if self.shuffle and len(batches) > 1:
            order = self.torch.randperm(len(batches)).tolist()
            batches = [batches[index] for index in order]
        yield from batches


def _random_swap(tokens, probability: float, ratio: float, window_size: int):
    if random.random() > float(probability) or len(tokens) <= 1:
        return tokens
    num_swaps = max(0, int(len(tokens) * float(ratio)))
    if num_swaps == 0:
        return tokens
    augmented = list(tokens)
    for _ in range(num_swaps):
        center = random.randint(0, len(augmented) - 1)
        window_start = max(0, center - int(window_size) // 2)
        window_end = min(
            len(augmented), center + int(window_size) // 2 + 1)
        if window_end - window_start >= 2:
            first, second = random.sample(range(window_start, window_end), 2)
            augmented[first], augmented[second] = (
                augmented[second], augmented[first])
    return augmented


def _sequence_mask(tokens, mask_token_id: int, probability: float, ratio: float):
    if random.random() > float(probability) or len(tokens) <= 1:
        return tokens
    count = max(1, min(int(len(tokens) * float(ratio)), len(tokens) - 1))
    positions = random.sample(range(len(tokens)), count)
    augmented = list(tokens)
    for position in positions:
        augmented[position] = int(mask_token_id)
    return augmented


def _augment_serialized_tokens(tokens, stage: str, mask_token_id: int):
    """Apply the pinned official pre-BPE token transforms."""
    tokens = list(tokens)
    if stage == "pretrain":
        return _random_swap(
            tokens, probability=0.5, ratio=0.1, window_size=3)
    if stage == "finetune":
        tokens = _random_swap(
            tokens, probability=0.4, ratio=0.05, window_size=3)
        return _sequence_mask(
            tokens, mask_token_id=mask_token_id,
            probability=0.3, ratio=0.05)
    if stage in {None, "evaluation"}:
        return tokens
    raise ValueError("augmentation stage must be pretrain, finetune or evaluation.")


def _make_loader(
        torch,
        encoded_split,
        batch_size: int,
        shuffle: bool,
        pad_token_id: int = 0,
        num_workers: int = 0,
        pin_memory: bool = False,
        bucket_by_length: bool = True,
        group_by_graph: bool = False,
        return_graph_ids: bool = False,
        tokenizer=None,
        augmentation_stage: str | None = None,
        max_sequence_length: int | None = None):
    input_ids = encoded_split["input_ids"]
    labels = encoded_split["labels"]
    if len(input_ids) != len(labels):
        raise ValueError("Encoded input and label counts must match.")
    if not input_ids:
        raise ValueError("Cannot create a paper DataLoader from an empty split.")
    graph_ids = list(encoded_split.get("graph_ids", range(len(input_ids))))
    if len(graph_ids) != len(input_ids):
        raise ValueError("Encoded graph ID and input counts must match.")
    serialized = list(encoded_split.get(
        "serialized_token_ids", [None] * len(input_ids)))
    if len(serialized) != len(input_ids):
        raise ValueError("Serialized token and input counts must match.")
    records = list(zip(input_ids, labels, graph_ids, serialized))
    if group_by_graph:
        grouped = {}
        for record in records:
            grouped.setdefault(int(record[2]), []).append(record)
        dataset = list(grouped.values())
        dataset_lengths = [
            max(len(record[0]) for record in group) for group in dataset]
    else:
        dataset = records
        dataset_lengths = [len(sequence) for sequence in input_ids]

    def collate_batch(batch):
        if group_by_graph:
            batch = [random.choice(group) for group in batch]
        prepared = []
        for sequence, label, graph_id, raw_tokens in batch:
            if augmentation_stage is not None:
                if tokenizer is None or raw_tokens is None:
                    raise ValueError(
                        "Token augmentation requires tokenizer and serialized tokens.")
                raw_tokens = _augment_serialized_tokens(
                    raw_tokens, augmentation_stage,
                    tokenizer.special_tokens.mask_token_id)
                sequence = tokenizer.encode_tokens(raw_tokens)
                if (
                        max_sequence_length is not None
                        and len(sequence) > int(max_sequence_length)):
                    sequence = [
                        *sequence[:int(max_sequence_length) - 1],
                        int(tokenizer.special_tokens.sep_token_id),
                    ]
            prepared.append((sequence, label, graph_id))
        batch = prepared
        maximum = max(len(sequence) for sequence, _, _ in batch)
        padded = torch.full(
            (len(batch), maximum), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros_like(padded)
        for row, (sequence, _, _) in enumerate(batch):
            length = len(sequence)
            if length:
                padded[row, :length] = torch.as_tensor(
                    sequence, dtype=torch.long)
                attention_mask[row, :length] = 1
        batch_labels = torch.as_tensor(
            [label for _, label, _ in batch], dtype=torch.float32)
        if return_graph_ids:
            return (
                padded, attention_mask, batch_labels,
                torch.as_tensor(
                    [graph_id for _, _, graph_id in batch], dtype=torch.long),
            )
        return padded, attention_mask, batch_labels

    workers = max(0, int(num_workers))
    worker_seed_generator = torch.Generator()
    worker_seed_generator.manual_seed(0)
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": bool(pin_memory),
        "collate_fn": collate_batch,
        "generator": worker_seed_generator,
    }
    if workers:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
        })
    if bucket_by_length:
        loader_kwargs["batch_sampler"] = _LengthBucketBatchSampler(
            torch,
            dataset_lengths,
            batch_size=int(batch_size),
            shuffle=shuffle,
        )
    else:
        loader_kwargs.update({
            "batch_size": int(batch_size),
            "shuffle": bool(shuffle),
        })
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
    loader.dynamic_padding = True
    loader.group_by_graph = bool(group_by_graph)
    loader.returns_graph_ids = bool(return_graph_ids)
    loader.worker_seed_generator = worker_seed_generator
    return loader


def _warmup_cosine_scheduler(torch, optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * float(warmup_ratio))
    if warmup_steps >= total_steps:
        warmup_steps = max(0, total_steps - 1)

    def multiplier(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.01 + 0.99 * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _mask_for_mlm(torch, input_ids, attention_mask, tokenizer, mask_prob: float):
    masked = input_ids.clone()
    labels = input_ids.clone()
    special = torch.zeros_like(labels, dtype=torch.bool)
    for token_id in (
        tokenizer.special_tokens.pad_token_id,
        tokenizer.special_tokens.cls_token_id,
        tokenizer.special_tokens.sep_token_id,
    ):
        special |= input_ids.eq(int(token_id))
    special |= ~attention_mask.bool()
    probabilities = torch.full(
        labels.shape, float(mask_prob), dtype=torch.float32,
        device=labels.device)
    probabilities.masked_fill_(special, 0.0)
    selected = torch.bernoulli(probabilities).bool()
    if not torch.any(selected):
        candidates = (~special).reshape(-1).nonzero(
            as_tuple=False).reshape(-1)
        if candidates.numel() == 0:
            raise ValueError("MLM batch contains no eligible tokens.")
        chosen = candidates[torch.randint(
            candidates.numel(), (1,), device=labels.device)]
        selected.reshape(-1)[chosen] = True
    labels[~selected] = -100
    replaced = torch.bernoulli(torch.full(
        labels.shape, 0.8, device=labels.device)).bool() & selected
    masked[replaced] = int(tokenizer.special_tokens.mask_token_id)
    random_replaced = (
        torch.bernoulli(torch.full(
            labels.shape, 0.5, device=labels.device)).bool()
        & selected & ~replaced)
    random_words = torch.randint(
        int(tokenizer.max_token_id) + 1, labels.shape,
        dtype=torch.long, device=labels.device)
    masked[random_replaced] = random_words[random_replaced]
    return masked, labels


def _loss(
        torch, logits, labels, spec, loss_name: str = "default",
        pos_weight=None, focal_gamma: float = 2.0):
    logits = logits.float()
    labels = labels.float()
    loss_name = str(loss_name).lower()
    if spec.canonical_name == "molhiv":
        if loss_name != "default" or pos_weight is not None:
            raise ValueError(
                "Strict OGBG-molhiv paper loss is unweighted cross entropy.")
        if logits.ndim != 2 or logits.shape[-1] != 2:
            raise ValueError("OGBG-molhiv paper model must output two logits.")
        labels = labels.reshape(-1)
        valid = torch.isfinite(labels)
        if not torch.any(valid):
            raise ValueError("OGBG-molhiv batch contains no valid labels.")
        return torch.nn.functional.cross_entropy(
            logits[valid], labels[valid].long())
    valid = None
    if spec.canonical_name == "peptides-struct":
        valid = ~torch.isnan(labels)
        if not torch.any(valid):
            raise ValueError("Peptides-struct batch contains no valid labels.")
        logits = logits[valid]
        labels = labels[valid]
    if loss_name == "default" and spec.canonical_name == "peptides-struct":
        return torch.nn.functional.l1_loss(logits, labels)
    if loss_name == "default":
        return torch.nn.functional.mse_loss(logits, labels)
    if loss_name == "l1":
        return torch.nn.functional.l1_loss(logits, labels)
    if loss_name == "huber":
        return torch.nn.functional.smooth_l1_loss(logits, labels)
    raise ValueError("Regression loss must be 'default', 'l1', or 'huber'.")


def _resolve_precision_options(torch, device, preset) -> Dict[str, Any]:
    amp_dtype = str(preset.get("amp_dtype", "off")).lower()
    if amp_dtype not in {"off", "fp16", "bf16"}:
        raise ValueError("amp_dtype must be 'off', 'fp16', or 'bf16'.")
    if amp_dtype != "off" and device.type != "cuda":
        raise ValueError("Mixed precision requires a CUDA device.")
    torch_dtype = {
        "off": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[amp_dtype]
    return {
        "amp_dtype": amp_dtype,
        "torch_dtype": torch_dtype,
        "grad_scaler": None,
        "allow_tf32": bool(preset.get("allow_tf32", False)),
    }


def _cuda_memory_query_device(torch, device):
    """Return an explicit CUDA index for APIs that reject bare ``cuda``."""
    if device.type == "cuda" and device.index is None:
        return torch.cuda.current_device()
    return device


def _visible_cuda_device(torch, device):
    logical_index = (
        torch.cuda.current_device()
        if device.index is None else int(device.index))
    visible_devices = [value.strip() for value in os.environ.get(
        "CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if logical_index < len(visible_devices):
        return visible_devices[logical_index]
    return str(logical_index)


def _wait_for_gpu_idle(torch, device, poll_seconds: int = 60) -> None:
    """Wait until the physical CUDA device has no active compute process."""
    if device.type != "cuda":
        return
    visible_device = _visible_cuda_device(torch, device)
    while True:
        result = subprocess.run(
            ["nvidia-smi", "-i", visible_device,
             "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True)
        active_pids = [line.strip() for line in result.stdout.splitlines()
                       if line.strip().isdigit()]
        if not active_pids:
            return
        print(json.dumps({
            "event": "paper_wait_for_gpu_idle",
            "device": visible_device,
            "active_pids": active_pids,
            "poll_seconds": int(poll_seconds),
        }), flush=True)
        time.sleep(int(poll_seconds))


def _new_grad_scaler(torch, amp_dtype: str):
    if str(amp_dtype).lower() != "fp16":
        return None
    return torch.cuda.amp.GradScaler(enabled=True)


def _autocast_context(torch, device, torch_dtype):
    if torch_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(
        device_type=device.type, dtype=torch_dtype, enabled=True)


def _set_paper_model_mode(model, training: bool) -> None:
    """Use TensorLayerX modes, with a PyTorch fallback for test doubles."""
    tlx_method = getattr(
        model, "set_train" if training else "set_eval", None)
    if callable(tlx_method):
        tlx_method()
    elif training:
        model.train()
    else:
        model.eval()


def _epoch_runtime_metrics(
        torch, device, num_examples: int, seconds: float) -> Dict[str, float]:
    seconds = float(seconds)
    metrics = {
        "seconds": seconds,
        "examples_per_second": float(num_examples) / max(seconds, 1e-12),
    }
    if device.type == "cuda":
        metrics["peak_cuda_memory_gib"] = (
            float(torch.cuda.max_memory_allocated(device)) / 1024 ** 3)
    return metrics


def _require_finite_tensor(torch, value, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {name} detected.")


def _require_finite_scalar(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite {name} detected.")
    return value


def _average_mae(y_true, y_pred):
    import numpy as np

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    task_scores = []
    for task_index in range(y_true.shape[1]):
        valid = ~np.isnan(y_true[:, task_index])
        if np.any(valid):
            task_scores.append(float(np.abs(
                y_true[valid, task_index] - y_pred[valid, task_index]).mean()))
    return sum(task_scores) / len(task_scores) if task_scores else float("nan")


def compute_paper_metric(spec, y_true, y_pred) -> float:
    import numpy as np

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if spec.canonical_name == "molhiv":
        try:
            from ogb.graphproppred import Evaluator
        except ImportError as error:
            raise RuntimeError("Paper OGBG-molhiv evaluation requires the ogb package.") from error
        evaluator = Evaluator(name="ogbg-molhiv")
        return float(evaluator.eval({
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        })["rocauc"])
    if spec.canonical_name == "peptides-struct":
        return _average_mae(y_true, y_pred)
    return float(np.abs(y_true - y_pred).mean())


def compute_paper_metric_details(
        spec, y_true, y_pred, target_names=None) -> Dict[str, Any]:
    import numpy as np

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    details = {"metric": compute_paper_metric(spec, y_true, y_pred)}
    if spec.canonical_name not in {"qm9", "peptides-struct"}:
        return details
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    configured_names = list(
        target_names if target_names is not None else getattr(spec, "label_keys", ()))
    if len(configured_names) == y_true.shape[1]:
        target_names = configured_names
    else:
        target_names = [f"target_{index}" for index in range(y_true.shape[1])]
    per_target = {}
    for index, target_name in enumerate(target_names):
        valid = np.isfinite(y_true[:, index]) & np.isfinite(y_pred[:, index])
        per_target[target_name] = (
            float(np.abs(
                y_true[valid, index] - y_pred[valid, index]).mean())
            if valid.any() else float("nan"))
    details["per_target_mae"] = per_target
    return details


def _evaluate(
        torch, model, loader, spec, device, normalizer, torch_dtype=None):
    _set_paper_model_mode(model, training=False)
    total_loss = 0.0
    total_examples = 0
    targets, predictions, graph_ids = [], [], []
    non_blocking = bool(getattr(loader, "pin_memory", False))
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = batch[:3]
            input_ids = input_ids.to(device, non_blocking=non_blocking)
            attention_mask = attention_mask.to(
                device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            with _autocast_context(torch, device, torch_dtype):
                logits = model(input_ids, attention_mask, task="supervised")
                _require_finite_tensor(torch, logits, "supervised logits")
                loss = _loss(torch, logits, labels, spec)
            _require_finite_tensor(torch, loss, "evaluation loss")
            total_loss += loss.item() * len(input_ids)
            total_examples += len(input_ids)
            metric_logits = logits.float()
            if spec.canonical_name == "molhiv":
                predictions.append(
                    torch.softmax(metric_logits, dim=-1)[:, 1:2].cpu())
            else:
                predictions.append(metric_logits.cpu())
            targets.append(labels.cpu())
            if len(batch) > 3:
                graph_ids.append(batch[3].cpu())
    y_true = torch.cat(targets, dim=0)
    y_pred = torch.cat(predictions, dim=0)
    num_sequences = int(len(y_true))
    if graph_ids:
        all_graph_ids = torch.cat(graph_ids, dim=0)
        unique_ids = []
        seen = set()
        for graph_id in all_graph_ids.tolist():
            if graph_id not in seen:
                seen.add(graph_id)
                unique_ids.append(graph_id)
        grouped_targets, grouped_predictions = [], []
        for graph_id in unique_ids:
            selected = all_graph_ids.eq(int(graph_id))
            grouped_targets.append(y_true[selected][0])
            grouped_predictions.append(y_pred[selected].mean(dim=0))
        y_true = torch.stack(grouped_targets)
        y_pred = torch.stack(grouped_predictions)
    average_loss = _require_finite_scalar(
        total_loss / max(1, total_examples), "evaluation loss")
    target_names = (
        normalizer.get("target_properties") if normalizer is not None else None)
    metric_details = compute_paper_metric_details(
        spec, y_true.numpy(), y_pred.numpy(), target_names=target_names)
    metric_details["metric_space"] = _metric_semantics(
        spec, normalizer)["metric_space"]
    if normalizer is not None and spec.canonical_name in {
            "qm9", "peptides-struct"}:
        raw_y_true = _denormalize_labels(torch, y_true, normalizer)
        raw_y_pred = _denormalize_labels(torch, y_pred, normalizer)
        raw_details = compute_paper_metric_details(
            spec, raw_y_true.numpy(), raw_y_pred.numpy(),
            target_names=target_names)
        metric_details = {
            **raw_details,
            "metric_space": _metric_semantics(spec, normalizer)["metric_space"],
        }
    metric = _require_finite_scalar(
        metric_details["metric"],
        f"{spec.canonical_name} validation metric",
    )
    return {
        "loss": average_loss,
        "metric": metric,
        "num_graphs": int(len(y_true)),
        "num_sequences": num_sequences,
        **{
            key: value for key, value in metric_details.items()
            if key != "metric"
        },
    }


def _rng_state(torch):
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _atomic_torch_save(torch, state, path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(state, temporary)
    os.replace(temporary, path)
    return str(path)


def _torch_load(torch, path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_paper_checkpoint(
        path, torch, model, epoch, best_metric, normalizer,
        experiment_fingerprint: str | None = None):
    """Save the small best-model artifact used only for final evaluation."""
    best_metric = _require_finite_scalar(best_metric, "checkpoint metric")
    state = {
        "checkpoint_kind": "best_model",
        "model": model.state_dict(),
        "epoch": int(epoch),
        "best_metric": best_metric,
        "normalizer": normalizer,
    }
    if experiment_fingerprint is not None:
        state["experiment_fingerprint"] = str(experiment_fingerprint)
    return _atomic_torch_save(torch, state, path)


def _validate_experiment_fingerprint(state, expected_fingerprint) -> None:
    if expected_fingerprint is None:
        return
    actual = state.get("experiment_fingerprint")
    if actual != str(expected_fingerprint):
        raise ValueError(
            "Checkpoint experiment fingerprint does not match the current "
            "data/tokenizer/configuration; use a new output directory or "
            "restart without --resume.")


def restore_paper_checkpoint(
        path, torch, model, device, expected_fingerprint=None):
    state = _torch_load(torch, path, device)
    if state.get("checkpoint_kind") != "best_model":
        raise ValueError(f"Not a best-model checkpoint: {path}")
    _validate_experiment_fingerprint(state, expected_fingerprint)
    model.load_state_dict(state["model"])
    return state


def save_paper_resume_state(
        path, torch, model, optimizer, scheduler, phase: str,
        epoch: int, extra: Dict[str, Any] | None = None, grad_scaler=None,
        experiment_fingerprint: str | None = None):
    state = {
        "checkpoint_kind": "resume_state",
        "phase": str(phase),
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": _rng_state(torch),
        "extra": dict(extra or {}),
    }
    if grad_scaler is not None:
        state["grad_scaler"] = grad_scaler.state_dict()
    if experiment_fingerprint is not None:
        state["experiment_fingerprint"] = str(experiment_fingerprint)
    return _atomic_torch_save(torch, state, path)


def _restore_rng_state(torch, state) -> None:
    rng_state = state.get("rng_state", {})
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])
    if "cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    if "numpy" in rng_state:
        try:
            import numpy as np

            np.random.set_state(rng_state["numpy"])
        except ImportError:
            pass


def restore_paper_resume_state(
        path, torch, model, optimizer, scheduler, device, grad_scaler=None,
        expected_fingerprint=None):
    state = _torch_load(torch, path, device)
    if state.get("checkpoint_kind") != "resume_state":
        raise ValueError(f"Not a resume-state checkpoint: {path}")
    _validate_experiment_fingerprint(state, expected_fingerprint)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    _restore_grad_scaler_state(state, grad_scaler)
    _restore_rng_state(torch, state)
    return state


def _restore_grad_scaler_state(state, grad_scaler) -> None:
    if grad_scaler is not None and "grad_scaler" in state:
        grad_scaler.load_state_dict(state["grad_scaler"])


def _experiment_fingerprint(
        preset, spec, encoded_cache_key: str, model_vocab_size: int,
        pooling: str, seed: int, run_index: int) -> str:
    payload = {
        "protocol_version": 3,
        "preset": preset,
        "dataset": spec.canonical_name,
        "task_type": spec.task_type,
        "output_dim": int(spec.output_dim),
        "encoded_cache_key": str(encoded_cache_key),
        "model_vocab_size": int(model_vocab_size),
        "pooling": str(pooling),
        "seed": int(seed),
        "run_index": int(run_index),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")).hexdigest()


def _optimizer_steps_per_epoch(loader, gradient_accumulation_steps: int) -> int:
    accumulation = max(1, int(gradient_accumulation_steps))
    return (len(loader) + accumulation - 1) // accumulation


def _molhiv_positive_weight(torch, encoded_train, device):
    labels = torch.as_tensor(encoded_train["labels"], dtype=torch.float32)
    labels = labels[torch.isfinite(labels)]
    positives = labels.eq(1).sum()
    negatives = labels.eq(0).sum()
    if positives.item() == 0 or negatives.item() == 0:
        raise ValueError(
            "OGBG-molhiv class weighting requires both classes in train.")
    return (negatives.float() / positives.float()).to(device)


def _supervised_loss_weight(torch, labels, spec) -> int:
    if spec.canonical_name == "peptides-struct":
        return int(torch.isfinite(labels).sum().item())
    return int(labels.numel())


def _normalize_accumulated_gradients(model, denominator: int) -> None:
    denominator = max(1, int(denominator))
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(denominator)


def _forward_supervised(
        torch, model, input_ids, attention_mask, gaussian_noise=None):
    if gaussian_noise is None or not callable(getattr(model, "_task_logits", None)):
        return model(input_ids, attention_mask, task="supervised")
    pooled = model(input_ids, attention_mask, task="pooled")
    if random.random() <= float(gaussian_noise["probability"]):
        pooled = pooled + torch.randn_like(pooled) * float(gaussian_noise["std"])
    return model._task_logits(pooled)


def _train_supervised(
        torch, model, loader, optimizer, scheduler, spec, device,
        max_grad_norm, gradient_accumulation_steps: int = 1,
        loss_name: str = "default", pos_weight=None,
        torch_dtype=None, grad_scaler=None, gaussian_noise=None):
    _set_paper_model_mode(model, training=True)
    non_blocking = bool(getattr(loader, "pin_memory", False))
    accumulation = max(1, int(gradient_accumulation_steps))
    total_batches = len(loader)
    total_loss = 0.0
    total_loss_weight = 0
    group_loss_weight = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (input_ids, attention_mask, labels) in enumerate(loader):
        labels = labels.to(device, non_blocking=non_blocking)
        with _autocast_context(torch, device, torch_dtype):
            logits = _forward_supervised(
                torch,
                model,
                input_ids.to(device, non_blocking=non_blocking),
                attention_mask.to(device, non_blocking=non_blocking),
                gaussian_noise=gaussian_noise,
            )
            _require_finite_tensor(torch, logits, "supervised logits")
            loss = _loss(
                torch,
                logits,
                labels,
                spec,
                loss_name=loss_name,
                pos_weight=pos_weight,
            )
        _require_finite_tensor(torch, loss, "supervised loss")
        loss_weight = _supervised_loss_weight(torch, labels, spec)
        total_loss += float(loss.detach().item()) * loss_weight
        total_loss_weight += loss_weight
        group_loss_weight += loss_weight
        scaled_loss = loss * loss_weight
        if grad_scaler is None:
            scaled_loss.backward()
        else:
            grad_scaler.scale(scaled_loss).backward()
        should_step = (step + 1) % accumulation == 0 or step + 1 == total_batches
        if not should_step:
            continue
        if grad_scaler is not None:
            grad_scaler.unscale_(optimizer)
        _normalize_accumulated_gradients(model, group_loss_weight)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        if grad_scaler is None:
            _require_finite_tensor(torch, grad_norm, "supervised gradient norm")
            optimizer.step()
            scheduler.step()
        else:
            scale_before_step = grad_scaler.get_scale()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            if grad_scaler.get_scale() >= scale_before_step:
                scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        group_loss_weight = 0
    return total_loss / max(1, total_loss_weight)


def _train_mlm(
        torch, model, loader, optimizer, scheduler, tokenizer, device,
        max_grad_norm, mask_prob, gradient_accumulation_steps: int = 1,
        torch_dtype=None, grad_scaler=None):
    _set_paper_model_mode(model, training=True)
    non_blocking = bool(getattr(loader, "pin_memory", False))
    accumulation = max(1, int(gradient_accumulation_steps))
    total_batches = len(loader)
    total_loss = 0.0
    total_loss_weight = 0
    group_loss_weight = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (input_ids, attention_mask, _) in enumerate(loader):
        masked_ids, labels = _mask_for_mlm(
            torch, input_ids, attention_mask, tokenizer, mask_prob)
        masked_ids = masked_ids.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        attention_mask = attention_mask.to(
            device, non_blocking=non_blocking)
        with _autocast_context(torch, device, torch_dtype):
            logits = model(masked_ids, attention_mask, task="mlm")
            _require_finite_tensor(torch, logits, "MLM logits")
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        _require_finite_tensor(torch, loss, "MLM loss")
        loss_weight = int(labels.ne(-100).sum().item())
        total_loss += float(loss.detach().item()) * loss_weight
        total_loss_weight += loss_weight
        group_loss_weight += loss_weight
        scaled_loss = loss * loss_weight
        if grad_scaler is None:
            scaled_loss.backward()
        else:
            grad_scaler.scale(scaled_loss).backward()
        should_step = (step + 1) % accumulation == 0 or step + 1 == total_batches
        if not should_step:
            continue
        if grad_scaler is not None:
            grad_scaler.unscale_(optimizer)
        _normalize_accumulated_gradients(model, group_loss_weight)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        if grad_scaler is None:
            _require_finite_tensor(torch, grad_norm, "MLM gradient norm")
            optimizer.step()
            scheduler.step()
        else:
            scale_before_step = grad_scaler.get_scale()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            if grad_scaler.get_scale() >= scale_before_step:
                scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        group_loss_weight = 0
    return total_loss / max(1, total_loss_weight)


def run_paper_experiment(args, spec, splits, fit_tokenizer):
    validate_paper_args(args, spec)
    torch = _torch()
    preset = paper_training_options(args, spec)
    validate_paper_training_options(preset, spec)
    seeds = paper_run_seeds(args)
    if not splits["train"]:
        raise ValueError("Paper protocol refuses an empty training split.")
    args.serialization = preset["serialization"]
    tokenizer = fit_tokenizer(args, splits["train"])
    cache_root = Path(
        getattr(args, "paper_cache_root", None)
        or (Path(args.data_root) / ".graph_tokenizer_cache")
    )
    encoded_cache_key = _encoded_splits_cache_key(
        tokenizer, splits, int(preset["max_position_embeddings"]),
        max_sequence_length=int(preset["max_sequence_length"]),
        num_realizations=int(preset["num_realizations"]))
    encoded_cache_path = (
        cache_root / spec.canonical_name / "encoded"
        / f"{encoded_cache_key}.pkl")
    encoded_cache_hit = encoded_cache_path.is_file()
    encoded, effective_max_length = _load_or_encode_splits(
        tokenizer,
        splits,
        int(preset["max_position_embeddings"]),
        encoded_cache_path,
        encoded_cache_key,
        max_sequence_length=int(preset["max_sequence_length"]),
        num_realizations=int(preset["num_realizations"]),
    )
    train_max_length = max(
        (len(sequence) for sequence in encoded["train"]["input_ids"]),
        default=1,
    )
    normalizer, output_dim = _prepare_labels(
        encoded, spec, getattr(args, "target_property", None))
    metric_semantics = _metric_semantics(spec, normalizer)
    model_vocab_size = tokenizer.max_token_id + 1
    tokenizer.validate_model_vocab(model_vocab_size)
    pooling = getattr(args, "pooling", "mean")
    device = torch.device(getattr(args, "device", "cuda"))
    if bool(getattr(args, "wait_for_gpu_idle", False)):
        _wait_for_gpu_idle(torch, device)
    precision = _resolve_precision_options(torch, device, preset)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = precision["allow_tf32"]
        torch.backends.cudnn.allow_tf32 = precision["allow_tf32"]
    if device.type == "cuda":
        memory_device = _cuda_memory_query_device(torch, device)
        try:
            available_bytes, _ = torch.cuda.mem_get_info(memory_device)
        except (AttributeError, TypeError):
            available_bytes = torch.cuda.get_device_properties(
                memory_device).total_memory
    else:
        available_bytes = 1 << 60
    memory_plan = _resolve_memory_plan(
        encoder_type=preset["encoder"],
        effective_batch_size=int(preset["batch_size"]),
        sequence_length=train_max_length,
        available_bytes=available_bytes,
        activation_bytes_per_element=(
            3 if precision["torch_dtype"] is not None else 4),
    )
    micro_batch_size = int(memory_plan["micro_batch_size"])
    accumulation_steps = int(memory_plan["gradient_accumulation_steps"])
    training_loss_name = str(preset.get("training_loss", "default"))
    molhiv_pos_weight = None
    if (
            spec.canonical_name == "molhiv"
            and bool(preset.get("molhiv_pos_weight", False))):
        molhiv_pos_weight = _molhiv_positive_weight(
            torch, encoded["train"], device)
    run_summaries = []
    experiment_fingerprints = []
    for run_index, seed in enumerate(seeds):
        set_paper_seed(seed, torch)
        experiment_fingerprint = _experiment_fingerprint(
            preset,
            spec,
            encoded_cache_key,
            model_vocab_size,
            pooling,
            seed=seed,
            run_index=run_index,
        )
        experiment_fingerprints.append(experiment_fingerprint)
        grad_scaler = _new_grad_scaler(torch, precision["amp_dtype"])
        model = _create_paper_model(
            encoder_type=preset["encoder"],
            vocab_size=model_vocab_size,
            pad_token_id=tokenizer.special_tokens.pad_token_id,
            task_type=spec.task_type,
            output_dim=output_dim,
            pooling=pooling,
            model_config={
                "max_position_embeddings": int(preset["max_position_embeddings"]),
            },
            allow_random_gte_init=bool(
                getattr(args, "allow_random_gte_init", False)),
            pretrained_cache_dir=getattr(args, "gte_cache_dir", None),
        ).to(device)
        _require_fp32_parameters(torch, model)
        loader_options = {
            "pad_token_id": tokenizer.special_tokens.pad_token_id,
            "num_workers": getattr(args, "num_workers", 0),
            "pin_memory": device.type == "cuda",
            "tokenizer": tokenizer,
            "max_sequence_length": int(preset["max_sequence_length"]),
        }
        pretrain_loader = _make_loader(
            torch, encoded["train"], micro_batch_size, shuffle=True,
            bucket_by_length=True, group_by_graph=True,
            augmentation_stage="pretrain", **loader_options)
        train_loader = _make_loader(
            torch, encoded["train"], micro_batch_size, shuffle=True,
            bucket_by_length=True, group_by_graph=True,
            augmentation_stage="finetune", **loader_options)
        val_loader = _make_loader(
            torch, encoded["val"], micro_batch_size, shuffle=False,
            bucket_by_length=False, group_by_graph=True, **loader_options)

        optimizer_steps = _optimizer_steps_per_epoch(
            train_loader, accumulation_steps)
        run_directory = (
            Path(args.output_dir) / "paper" / spec.canonical_name
            / preset["encoder"] / f"run_{run_index}")
        checkpoint_path = run_directory / "best.pt"
        resume_path = run_directory / "last_state.pt"
        resume_interval = max(1, int(preset.get("resume_interval", 5)))
        resume_state = None
        resume_phase = None
        if bool(getattr(args, "resume", False)) and resume_path.is_file():
            resume_state = _torch_load(torch, resume_path, device)
            if resume_state.get("checkpoint_kind") != "resume_state":
                raise ValueError(f"Not a resume-state checkpoint: {resume_path}")
            _validate_experiment_fingerprint(
                resume_state, experiment_fingerprint)
            resume_phase = resume_state.get("phase")
            model.load_state_dict(resume_state["model"])

        pretrain_epochs = int(preset["pretrain_epochs"])
        if resume_phase not in {
                "pretrain_complete", "finetune", "finetune_complete"}:
            pretrain_optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(preset["pretrain_lr"]),
                weight_decay=float(preset["weight_decay"]),
            )
            pretrain_scheduler = _warmup_cosine_scheduler(
                torch, pretrain_optimizer,
                _optimizer_steps_per_epoch(
                    pretrain_loader, accumulation_steps) * pretrain_epochs,
                preset["pretrain_warmup_ratio"])
            pretrain_start = 1
            if resume_phase == "pretrain":
                resumed = restore_paper_resume_state(
                    resume_path, torch, model, pretrain_optimizer,
                    pretrain_scheduler, device,
                    grad_scaler=grad_scaler,
                    expected_fingerprint=experiment_fingerprint)
                pretrain_start = int(resumed["epoch"]) + 1
            for epoch in range(pretrain_start, pretrain_epochs + 1):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                started = time.monotonic()
                mlm_loss = _train_mlm(
                    torch, model, pretrain_loader, pretrain_optimizer,
                    pretrain_scheduler, tokenizer, device,
                    preset["pretrain_max_grad_norm"], preset["mask_prob"],
                    gradient_accumulation_steps=accumulation_steps,
                    torch_dtype=precision["torch_dtype"],
                    grad_scaler=grad_scaler)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                runtime_metrics = _epoch_runtime_metrics(
                    torch, device, len(encoded["train"]),
                    time.monotonic() - started)
                print(json.dumps({
                    "event": "paper_epoch",
                    "phase": "pretrain",
                    "dataset": spec.canonical_name,
                    "model": preset["encoder"],
                    "run": run_index,
                    "seed": seed,
                    "epoch": epoch,
                    "epochs": pretrain_epochs,
                    "loss": mlm_loss,
                    **runtime_metrics,
                }), flush=True)
                if epoch % resume_interval == 0 or epoch == pretrain_epochs:
                    save_paper_resume_state(
                        resume_path, torch, model, pretrain_optimizer,
                        pretrain_scheduler, phase="pretrain", epoch=epoch,
                        grad_scaler=grad_scaler,
                        experiment_fingerprint=experiment_fingerprint)
            save_paper_resume_state(
                resume_path, torch, model, pretrain_optimizer,
                pretrain_scheduler, phase="pretrain_complete",
                epoch=pretrain_epochs,
                grad_scaler=grad_scaler,
                experiment_fingerprint=experiment_fingerprint)
        elif resume_state is not None:
            _restore_rng_state(torch, resume_state)
            _restore_grad_scaler_state(resume_state, grad_scaler)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(preset["finetune_lr"]),
            weight_decay=float(preset["weight_decay"]),
        )
        finetune_epochs = int(preset["finetune_epochs"])
        scheduler = _warmup_cosine_scheduler(
            torch, optimizer,
            optimizer_steps * finetune_epochs,
            preset["finetune_warmup_ratio"])
        history = []
        best_metric = None
        stale_epochs = 0
        finetune_start = 1
        finetune_checkpoint_complete = resume_phase == "finetune_complete"
        skip_finetuning = finetune_checkpoint_complete
        if resume_phase in {"finetune", "finetune_complete"}:
            resumed = restore_paper_resume_state(
                resume_path, torch, model, optimizer, scheduler, device,
                grad_scaler=grad_scaler,
                expected_fingerprint=experiment_fingerprint)
            finetune_start = int(resumed["epoch"]) + 1
            extra = dict(resumed.get("extra", {}))
            history = list(extra.get("history", []))
            best_metric = extra.get("best_metric")
            stale_epochs = int(extra.get("stale_epochs", 0))
            skip_finetuning = _should_skip_finetuning(
                resume_phase, stale_epochs, int(preset["patience"]))
            if skip_finetuning:
                finetune_start = finetune_epochs + 1
        higher_is_better = spec.canonical_name == "molhiv"
        final_training_epoch = (
            int(resumed["epoch"])
            if skip_finetuning else finetune_start - 1)
        for epoch in range(finetune_start, finetune_epochs + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.monotonic()
            training_loss = _train_supervised(
                torch, model, train_loader, optimizer, scheduler, spec, device,
                preset["finetune_max_grad_norm"],
                gradient_accumulation_steps=accumulation_steps,
                loss_name=training_loss_name,
                pos_weight=molhiv_pos_weight,
                torch_dtype=precision["torch_dtype"],
                grad_scaler=grad_scaler,
                gaussian_noise=preset["gaussian_noise"])
            validation = _evaluate(
                torch, model, val_loader, spec, device, normalizer,
                torch_dtype=precision["torch_dtype"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            runtime_metrics = _epoch_runtime_metrics(
                torch, device, len(encoded["train"]),
                time.monotonic() - started)
            history.append({"epoch": epoch, "val_metric": validation["metric"], "val_loss": validation["loss"]})
            final_training_epoch = epoch
            improved = best_metric is None or (
                validation["metric"] > best_metric
                if higher_is_better else validation["metric"] < best_metric
            )
            if improved:
                best_metric = validation["metric"]
                stale_epochs = 0
                save_paper_checkpoint(
                    checkpoint_path, torch, model, epoch, best_metric,
                    normalizer,
                    experiment_fingerprint=experiment_fingerprint)
            else:
                stale_epochs += 1
            print(json.dumps({
                "event": "paper_epoch",
                "phase": "finetune",
                "dataset": spec.canonical_name,
                "model": preset["encoder"],
                "run": run_index,
                "seed": seed,
                "epoch": epoch,
                "epochs": finetune_epochs,
                "training_loss": training_loss,
                "val_loss": validation["loss"],
                "val_metric": validation["metric"],
                "best_metric": best_metric,
                "stale_epochs": stale_epochs,
                **runtime_metrics,
            }), flush=True)
            should_stop = stale_epochs >= int(preset["patience"])
            if (
                    epoch % resume_interval == 0
                    or epoch == finetune_epochs
                    or should_stop):
                save_paper_resume_state(
                    resume_path, torch, model, optimizer, scheduler,
                    phase="finetune", epoch=epoch, extra={
                        "history": history,
                        "best_metric": best_metric,
                        "stale_epochs": stale_epochs,
                        "normalizer": normalizer,
                    }, grad_scaler=grad_scaler,
                    experiment_fingerprint=experiment_fingerprint)
            if should_stop:
                break
        if not finetune_checkpoint_complete:
            save_paper_resume_state(
                resume_path, torch, model, optimizer, scheduler,
                phase="finetune_complete", epoch=final_training_epoch, extra={
                    "history": history,
                    "best_metric": best_metric,
                    "stale_epochs": stale_epochs,
                    "normalizer": normalizer,
                }, grad_scaler=grad_scaler,
                experiment_fingerprint=experiment_fingerprint)
        state = restore_paper_checkpoint(
            checkpoint_path, torch, model, device,
            expected_fingerprint=experiment_fingerprint)
        final_validation = _evaluate(
            torch, model, val_loader, spec, device, state["normalizer"],
            torch_dtype=precision["torch_dtype"])
        test_loader = _make_loader(
            torch, encoded["test"], micro_batch_size, shuffle=False,
            bucket_by_length=False, return_graph_ids=True, **loader_options)
        final_test = _evaluate(
            torch, model, test_loader, spec, device, state["normalizer"],
            torch_dtype=precision["torch_dtype"])
        run_summaries.append({
            "run": run_index,
            "seed": seed,
            "best_epoch": state["epoch"],
            "best_val": final_validation,
            "best_test": final_test,
            "checkpoint_path": str(checkpoint_path),
            "resume_checkpoint_path": str(resume_path),
            "resumed": resume_state is not None,
            "history": history,
            "test_evaluations": 1,
            "experiment_fingerprint": experiment_fingerprint,
            "model": model.manifest(),
            **metric_semantics,
        })
    return {
        "dataset": spec.canonical_name,
        "model": preset["encoder"],
        "protocol": "paper",
        "runs": run_summaries,
        "preset": preset,
        "effective_max_length": effective_max_length,
        "tokenizer_vocab_size": tokenizer.max_token_id + 1,
        "experiment_fingerprints": experiment_fingerprints,
        "memory_plan": memory_plan,
        "precision": {
            "amp_dtype": precision["amp_dtype"],
            "allow_tf32": precision["allow_tf32"],
        },
        "training_loss": {
            "name": training_loss_name,
            "molhiv_pos_weight": (
                float(molhiv_pos_weight.item())
                if molhiv_pos_weight is not None else None),
        },
        **metric_semantics,
        "cache": {
            "tokenizer_status": getattr(tokenizer, "_cache_status", "unknown"),
            "tokenizer_path": getattr(tokenizer, "_cache_path", None),
            "encoded_status": "hit" if encoded_cache_hit else "miss",
            "encoded_path": str(encoded_cache_path),
        },
    }
