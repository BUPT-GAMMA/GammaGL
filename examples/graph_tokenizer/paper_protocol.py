"""Torch-only training protocol aligned with the GraphTokenizer paper."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


PAPER_RUNTIME_REQUIREMENTS = {
    "torch": "2.1.2",
    "cuda": "12.1",
    "dgl": "2.4.0",
    "torch_geometric": "2.4.0",
}

PAPER_ARCHITECTURES = {
    "bert": {
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "parameter_estimate": 16_000_000,
    },
    "gte": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
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


def _load_model_module(module_name: str, filename: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = Path(__file__).resolve().parents[2] / "gammagl" / "models" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_paper_model_classes():
    graph_bert = _load_model_module(
        "graph_tokenizer_native_graph_bert", "graph_bert.py")
    # graph_gte.py retains a direct-file import fallback for focused tests.
    sys.modules["graph_bert"] = graph_bert
    graph_gte = _load_model_module(
        "graph_tokenizer_native_graph_gte", "graph_gte.py")
    return graph_bert.GraphBERT, graph_gte.GraphGTE


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
        strict_architecture: bool = True):
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
    return model_class(**kwargs)


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
        if target_property is not None:
            raise ValueError(
                "QM9 strict paper protocol is a joint 16-target regression; "
                "do not pass --target-property.")
    requested_model = getattr(args, "model", None)
    if requested_model and requested_model not in {"bert", "gte"}:
        raise ValueError("Paper protocol model must be 'bert' or 'gte'.")


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


def paper_training_options(args) -> Dict[str, Any]:
    options = {
        "encoder": args.model,
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
        encoding_batch_size: int = 2048):
    encoded = {}
    global_max_length = 1
    for split_name, graphs in splits.items():
        if not graphs:
            raise ValueError(
                f"Paper protocol requires a non-empty {split_name} split.")
        if hasattr(tokenizer, "batch_encode_graphs"):
            sequences = []
            encoding_batch_size = max(1, int(encoding_batch_size))
            for start in range(0, len(graphs), encoding_batch_size):
                graph_batch = graphs[start:start + encoding_batch_size]
                results = tokenizer.batch_encode_graphs(graph_batch)
                if len(results) != len(graph_batch):
                    raise RuntimeError(
                        "Tokenizer batch encoding returned the wrong count.")
                sequences.extend(result.input_ids for result in results)
        else:
            sequences = [
                tokenizer.encode_graph(graph).input_ids for graph in graphs
            ]
        maximum = max((len(sequence) for sequence in sequences), default=1)
        if maximum > max_position_embeddings:
            raise ValueError(
                f"{split_name} sequence length {maximum} exceeds "
                f"max_position_embeddings={max_position_embeddings}.")
        global_max_length = max(global_max_length, maximum)
        encoded[split_name] = {
            "input_ids": [list(map(int, sequence)) for sequence in sequences],
            "labels": [list(graph.y) for graph in graphs],
        }
    return encoded, global_max_length


def _load_or_encode_splits(
        tokenizer, splits, max_position_embeddings: int,
        cache_path, cache_key: str):
    cache_path = Path(cache_path)
    if cache_path.is_file():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if (
                    cached.get("cache_version") == 1
                    and cached.get("cache_key") == str(cache_key)
                    and cached.get("max_position_embeddings")
                    == int(max_position_embeddings)):
                return cached["encoded"], int(cached["global_max_length"])
        except (OSError, EOFError, AttributeError, KeyError, pickle.UnpicklingError):
            pass
    encoded, global_max_length = _encode_splits(
        tokenizer, splits, int(max_position_embeddings))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(
        f".{cache_path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump({
            "cache_version": 1,
            "cache_key": str(cache_key),
            "max_position_embeddings": int(max_position_embeddings),
            "global_max_length": int(global_max_length),
            "encoded": encoded,
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, cache_path)
    return encoded, global_max_length


def _encoded_splits_cache_key(tokenizer, splits, max_position_embeddings: int) -> str:
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
        "cache_version": 1,
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
    }
    return hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest()


def _prepare_labels(encoded, spec, target_property):
    if spec.canonical_name == "qm9":
        output_dim = int(spec.output_dim)
        target_properties = list(spec.label_keys)
        if len(target_properties) != output_dim:
            raise ValueError(
                f"QM9 requires exactly {output_dim} target property names; "
                f"received {len(target_properties)}.")
        if not encoded["train"]["labels"]:
            raise ValueError("QM9 training split cannot be empty.")
        for split_name, split in encoded.items():
            for row_index, row in enumerate(split["labels"]):
                if len(row) != output_dim:
                    raise ValueError(
                        f"QM9 {split_name} label row {row_index} must contain "
                        f"exactly {output_dim} targets; received {len(row)}.")
                if not all(math.isfinite(float(value)) for value in row):
                    raise ValueError(
                        f"QM9 {split_name} label row {row_index} contains a "
                        "non-finite target.")
        train_rows = [list(map(float, row)) for row in encoded["train"]["labels"]]
        means = [
            sum(row[index] for row in train_rows) / len(train_rows)
            for index in range(output_dim)
        ]
        variances = [
            sum((row[index] - means[index]) ** 2 for row in train_rows)
            / len(train_rows)
            for index in range(output_dim)
        ]
        stds = [max(math.sqrt(variance), 1e-12) for variance in variances]
        for split in encoded.values():
            split["labels"] = [
                [
                    (float(row[index]) - means[index]) / stds[index]
                    for index in range(output_dim)
                ]
                for row in split["labels"]
            ]
        return {
            "mean": means,
            "std": stds,
            "target_properties": target_properties,
        }, output_dim
    return None, spec.output_dim


def _denormalize_labels(torch, values, normalizer):
    means = torch.as_tensor(
        normalizer["mean"], dtype=values.dtype, device=values.device)
    stds = torch.as_tensor(
        normalizer["std"], dtype=values.dtype, device=values.device)
    if values.ndim < 1 or values.shape[-1] != len(means):
        raise ValueError(
            "Normalized label width does not match the QM9 normalizer.")
    return values * stds + means


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


def _make_loader(
        torch,
        encoded_split,
        batch_size: int,
        shuffle: bool,
        pad_token_id: int = 0,
        num_workers: int = 0,
        pin_memory: bool = False,
        bucket_by_length: bool = True):
    input_ids = encoded_split["input_ids"]
    labels = encoded_split["labels"]
    if len(input_ids) != len(labels):
        raise ValueError("Encoded input and label counts must match.")
    if not input_ids:
        raise ValueError("Cannot create a paper DataLoader from an empty split.")
    dataset = list(zip(input_ids, labels))

    def collate_batch(batch):
        maximum = max(len(sequence) for sequence, _ in batch)
        padded = torch.full(
            (len(batch), maximum), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros_like(padded)
        for row, (sequence, _) in enumerate(batch):
            length = len(sequence)
            if length:
                padded[row, :length] = torch.as_tensor(
                    sequence, dtype=torch.long)
                attention_mask[row, :length] = 1
        batch_labels = torch.as_tensor(
            [label for _, label in batch], dtype=torch.float32)
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
            [len(sequence) for sequence in input_ids],
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
    loader.worker_seed_generator = worker_seed_generator
    return loader


def _linear_warmup_scheduler(torch, optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * float(warmup_ratio)))

    def multiplier(step):
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _mask_for_mlm(torch, input_ids, attention_mask, tokenizer, mask_prob: float):
    labels = input_ids.clone()
    maskable = attention_mask.bool()
    for token_id in (
        tokenizer.special_tokens.pad_token_id,
        tokenizer.special_tokens.cls_token_id,
        tokenizer.special_tokens.sep_token_id,
        tokenizer.special_tokens.component_sep_token_id,
    ):
        maskable &= input_ids.ne(int(token_id))
    selected = torch.rand_like(input_ids, dtype=torch.float32).lt(float(mask_prob)) & maskable
    if float(mask_prob) > 0:
        flat_maskable = maskable.reshape(-1)
        first_maskable = flat_maskable & flat_maskable.cumsum(dim=0).eq(1)
        selected |= first_maskable.reshape_as(selected) & ~selected.any()
    labels[~selected] = -100
    masked = input_ids.clone()
    masked[selected] = int(tokenizer.special_tokens.mask_token_id)
    return masked, labels


def _loss(
        torch, logits, labels, spec, loss_name: str = "default",
        pos_weight=None, focal_gamma: float = 2.0):
    logits = logits.float()
    labels = labels.float()
    loss_name = str(loss_name).lower()
    if spec.canonical_name == "molhiv":
        if loss_name not in {"default", "focal"}:
            raise ValueError("OGBG-molhiv loss must be 'default' or 'focal'.")
        logits = logits.view(-1)
        labels = labels.view(-1)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight, reduction="none")
        if loss_name == "focal":
            probabilities = torch.sigmoid(logits)
            probability_of_target = (
                probabilities * labels + (1 - probabilities) * (1 - labels))
            bce = (1 - probability_of_target).pow(float(focal_gamma)) * bce
        return bce.mean()
    valid = None
    if spec.canonical_name == "peptides-struct":
        valid = ~torch.isnan(labels)
        if not torch.any(valid):
            raise ValueError("Peptides-struct batch contains no valid labels.")
        logits = logits[valid]
        labels = labels[valid]
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


def compute_paper_metric_details(spec, y_true, y_pred) -> Dict[str, Any]:
    import numpy as np

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    details = {"metric": compute_paper_metric(spec, y_true, y_pred)}
    if spec.canonical_name not in {"qm9", "peptides-struct"}:
        return details
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    configured_names = list(getattr(spec, "label_keys", ()))
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
    targets, predictions = [], []
    non_blocking = bool(getattr(loader, "pin_memory", False))
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
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
                predictions.append(torch.sigmoid(metric_logits).cpu())
            else:
                predictions.append(metric_logits.cpu())
            targets.append(labels.cpu())
    y_true = torch.cat(targets, dim=0)
    y_pred = torch.cat(predictions, dim=0)
    average_loss = _require_finite_scalar(
        total_loss / max(1, total_examples), "evaluation loss")
    metric_details = compute_paper_metric_details(
        spec, y_true.numpy(), y_pred.numpy())
    if normalizer is not None and spec.canonical_name == "qm9":
        raw_y_true = _denormalize_labels(torch, y_true, normalizer)
        raw_y_pred = _denormalize_labels(torch, y_pred, normalizer)
        raw_details = compute_paper_metric_details(
            spec, raw_y_true.numpy(), raw_y_pred.numpy())
        metric_details["metric"] = raw_details["metric"]
        metric_details["metric_space"] = "raw"
        metric_details["per_target_mae_raw"] = raw_details["per_target_mae"]
    metric = _require_finite_scalar(
        metric_details["metric"],
        f"{spec.canonical_name} validation metric",
    )
    return {
        "loss": average_loss,
        "metric": metric,
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
        "protocol_version": 2,
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


def _train_supervised(
        torch, model, loader, optimizer, scheduler, spec, device,
        max_grad_norm, gradient_accumulation_steps: int = 1,
        loss_name: str = "default", pos_weight=None,
        torch_dtype=None, grad_scaler=None):
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
            logits = model(
                input_ids.to(device, non_blocking=non_blocking),
                attention_mask.to(device, non_blocking=non_blocking),
                task="supervised",
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
        input_ids = input_ids.to(device, non_blocking=non_blocking)
        attention_mask = attention_mask.to(
            device, non_blocking=non_blocking)
        masked_ids, labels = _mask_for_mlm(torch, input_ids, attention_mask, tokenizer, mask_prob)
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
    preset = paper_training_options(args)
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
        tokenizer, splits, int(preset["max_position_embeddings"]))
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
    )
    normalizer, output_dim = _prepare_labels(
        encoded, spec, getattr(args, "target_property", None))
    model_vocab_size = tokenizer.max_token_id + 1
    tokenizer.validate_model_vocab(model_vocab_size)
    pooling = getattr(args, "pooling", "mean")
    device = torch.device(getattr(args, "device", "cuda"))
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
        sequence_length=effective_max_length,
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
                "max_position_embeddings": effective_max_length,
            },
        ).to(device)
        _require_fp32_parameters(torch, model)
        loader_options = {
            "pad_token_id": tokenizer.special_tokens.pad_token_id,
            "num_workers": getattr(args, "num_workers", 0),
            "pin_memory": device.type == "cuda",
        }
        train_loader = _make_loader(
            torch, encoded["train"], micro_batch_size, shuffle=True,
            bucket_by_length=True, **loader_options)
        val_loader = _make_loader(
            torch, encoded["val"], micro_batch_size, shuffle=False,
            bucket_by_length=False, **loader_options)
        test_loader = _make_loader(
            torch, encoded["test"], micro_batch_size, shuffle=False,
            bucket_by_length=False, **loader_options)

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
            pretrain_scheduler = _linear_warmup_scheduler(
                torch, pretrain_optimizer,
                optimizer_steps * pretrain_epochs,
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
                    torch, model, train_loader, pretrain_optimizer,
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
        scheduler = _linear_warmup_scheduler(
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
                grad_scaler=grad_scaler)
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
        "cache": {
            "tokenizer_status": getattr(tokenizer, "_cache_status", "unknown"),
            "tokenizer_path": getattr(tokenizer, "_cache_path", None),
            "encoded_status": "hit" if encoded_cache_hit else "miss",
            "encoded_path": str(encoded_cache_path),
        },
    }
