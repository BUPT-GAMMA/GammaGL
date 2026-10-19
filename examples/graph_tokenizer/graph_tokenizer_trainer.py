"""GraphTokenizer training entrypoint.

The trainer deliberately owns only the train/validation/test pipeline.  It
uses GammaGL's serializers, BPE implementation, tokenizer, datasets, and
native TensorLayerX GraphBERT/GraphGTE models; it does not orchestrate paper
benchmarks. Formal GTE presets select official random initialization for AQSOL
and the native HF-to-TLX pretrained converter for the two TU datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import statistics
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

# A direct CLI invocation must select TF devices before importing TLX, whose
# imports may initialize TensorFlow. Programmatic entry points also configure
# before seeds/data below and reject an incompatible preinitialized runtime.
_SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
if _SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _SOURCE_ROOT)
from examples.graph_tokenizer.runtime import configure_device
from examples.graph_tokenizer.artifacts import ARTIFACT_VERSION, atomic_write

if __name__ == '__main__' and os.environ.get('TL_BACKEND') == 'tensorflow':
    if not any(flag in sys.argv[1:] for flag in ('--help', '-h', '--prepare-data')):
        _device_parser = argparse.ArgumentParser(add_help=False)
        _device_parser.add_argument('--device', default='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
        _device_args, _ = _device_parser.parse_known_args()
        configure_device('tensorflow', _device_args.device)

import tensorlayerx as tlx


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: tuple[str, ...]
    dataset_class: str
    storage_name: str
    task_type: str
    label_width: int
    num_classes: int
    metric: str
    higher_is_better: bool
    label_names: tuple[str, ...]

    @property
    def output_dim(self) -> int:
        return self.num_classes or self.label_width


DATASETS = (
    DatasetSpec("mutagenicity", ("mutagenicity", "mutag", "muta"),
                "Mutagenicity", "mutagenicity", "binary_classification", 1, 2,
                "Accuracy", True, ("label",)),
    DatasetSpec("dblp", ("dblp", "dblp-v1", "dblp_v1"),
                "GraphTokenizerDBLP", "dblp", "binary_classification", 1, 2,
                "Accuracy", True, ("label",)),
    DatasetSpec("aqsol", ("aqsol",), "AQSOL", "aqsol", "regression", 1, 0,
                "MAE", False, ("solubility",)),
)

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
PREPROCESSING_CACHE_VERSION = 9


# GammaGL presets derived from the pinned official GraphTokenizer implementation.
# CLI values override every item below.  ``max_length`` is the final input
# limit, distinct from the model's ``max_position_embeddings`` capacity.
_BASE_CONFIG = {
    "fit_scope": "all",
    "token_protocol": "official",
    "bpe_encode_mode": "all",
    "pretrain_batch_size": 32,
    "finetune_batch_size": 32,
    "learning_rate": 1e-5,
    "pretrain_learning_rate": 1e-4,
    "pretrain_epochs": 200,
    "finetune_epochs": 200,
    "max_length": 8096,
    "max_position_embeddings": 8096,
    "bpe_merges": 2000,
    "bpe_min_frequency": 2,
    "num_serializations": 100,
    "early_stopping_patience": 20,
    "mask_probability": 0.09,
    "pretrain_weight_decay": 0.1,
    "finetune_weight_decay": 0.1,
    "pretrain_warmup_ratio": 0.12,
    "finetune_warmup_ratio": 0.025,
    "pretrain_max_grad_norm": 2.0,
    "finetune_max_grad_norm": 0.5,
    "pretrain_swap_probability": 0.5,
    "pretrain_swap_ratio": 0.10,
    "pretrain_swap_window": 3,
    "finetune_swap_probability": 0.4,
    "finetune_swap_ratio": 0.05,
    "finetune_swap_window": 3,
    "finetune_mask_probability": 0.3,
    "finetune_mask_ratio": 0.05,
    "finetune_noise_probability": 0.3,
    "finetune_noise_std": 0.01,
    "finetune_use_consistency_regularization": False,
    "finetune_consistency_alpha": 1.0,
    "gradient_accumulation_steps": 1,
    "pooling": "mean",
    "gte_reset_weights": True,
}
PAPER_CONFIGS = {
    ("mutagenicity", "bert"): {**_BASE_CONFIG, "serializer": "feuler"},
    ("mutagenicity", "gte"): {
        **_BASE_CONFIG, "serializer": "feuler", "bpe_encode_mode": "none",
        "finetune_batch_size": 64, "learning_rate": 5e-5,
        "finetune_epochs": 100, "finetune_weight_decay": 0.01,
        "early_stopping_patience": 10, "max_position_embeddings": 8192,
        "finetune_use_consistency_regularization": True,
        "gte_reset_weights": False,
    },
    ("dblp", "bert"): {**_BASE_CONFIG, "serializer": "feuler"},
    ("dblp", "gte"): {
        **_BASE_CONFIG, "serializer": "eulerian", "bpe_encode_mode": "none",
        "finetune_batch_size": 64, "learning_rate": 5e-5,
        "finetune_epochs": 100, "finetune_weight_decay": 0.01,
        "early_stopping_patience": 10, "max_position_embeddings": 8192,
        "finetune_use_consistency_regularization": True,
        "gte_reset_weights": False,
    },
    ("aqsol", "bert"): {**_BASE_CONFIG, "serializer": "feuler"},
    ("aqsol", "gte"): {**_BASE_CONFIG, "serializer": "feuler", "max_position_embeddings": 8192},
}


class GraphRecord:
    def __init__(self, edge_index, x, edge_attr, y):
        self.edge_index, self.x, self.edge_attr, self.y = edge_index, x, edge_attr, y
        self.num_nodes = len(x)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_repo_on_path() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_dataset(name: str) -> DatasetSpec:
    normalized = str(name).lower().replace("_", "-")
    for spec in DATASETS:
        if normalized in {item.replace("_", "-") for item in spec.aliases}:
            return spec
    choices = ', '.join(spec.name for spec in DATASETS)
    raise ValueError(f"dataset must be one of: {choices}")


def ensure_tlx_backend():
    if tlx.BACKEND not in ("torch", "tensorflow"):
        raise RuntimeError("GraphTokenizer training supports TLX torch and TensorFlow eager backends.")
    if tlx.BACKEND == "tensorflow":
        import tensorflow as tf
        if not tf.executing_eagerly():
            raise RuntimeError("GraphTokenizer TensorFlow training requires eager execution.")
    return tlx


def training_helpers():
    ensure_repo_on_path()
    from examples.graph_tokenizer import tlx_training
    return tlx_training


def move_tensor(value, device):
    if tlx.BACKEND == "torch":
        return value.to(device)
    return value


def make_optimizer(lr, weight_decay):
    optimizer = tlx.optimizers.Adam(lr=lr, eps=1e-8)
    optimizer.graph_tokenizer_weight_decay = weight_decay
    return optimizer


def set_seed(backend, seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    if backend is tlx:
        tlx.set_seed(seed)
        if tlx.BACKEND != "torch":
            return
        import torch as backend
    backend.manual_seed(seed)
    if backend.cuda.is_available():
        backend.cuda.manual_seed_all(seed)
    cudnn = getattr(getattr(backend, "backends", None), "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = True
        cudnn.benchmark = False


def build_warmup_cosine_scheduler(
        backend, optimizer, total_steps: int, warmup_ratio: float, min_lr_ratio: float = 0.01):
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1).")
    if not 0.0 < min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in (0, 1].")
    warmup_steps = int(total_steps * warmup_ratio)

    def multiplier(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    if hasattr(optimizer, "apply_gradients"):
        return training_helpers().WarmupCosineScheduler(optimizer, multiplier)
    return backend.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def emit_event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def require_finite_tensor(backend, value, context):
    array = value if isinstance(value, np.ndarray) else tlx.convert_to_numpy(value)
    if not np.isfinite(array).all():
        raise FloatingPointError(f"Non-finite {context}")


def require_finite_metrics(value, context):
    """Validate numeric result fields without changing their values."""
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_metrics(item, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_metrics(item, f"{context}[{index}]")
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise FloatingPointError(f"Non-finite {context}: {value}")


def validate_run_result(result):
    context = f"run seed={result.get('seed')} run_index={result.get('run_index')}"
    if (result.get("complete") is False or result.get("success") is False
            or result.get("status") in ("failed", "error") or result.get("error")):
        raise ValueError(f"Cannot aggregate failed {context}")
    require_finite_metrics(result, context)


def to_list(value):
    for method in ("detach", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value.tolist() if hasattr(value, "tolist") else value


def is_sequence(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict))


def flatten_feature_ids(values) -> List[int]:
    if values is None:
        return []
    result = []
    for value in to_list(values):
        if is_sequence(value):
            if len(value) != 1 or is_sequence(value[0]):
                raise ValueError("Multi-dimensional features need a dataset-specific token adapter.")
            value = value[0]
        result.append(int(value))
    return result


def graph_label(value, width: int, allow_nan: bool) -> List[float]:
    values = to_list(value)
    while is_sequence(values) and len(values) == 1 and is_sequence(values[0]):
        values = values[0]
    if not is_sequence(values):
        values = [values]
    if len(values) != width:
        raise ValueError(f"Expected {width} labels, received {len(values)}.")
    result = [float(item) for item in values]
    if not allow_nan and not all(math.isfinite(item) for item in result):
        raise ValueError("Labels must be finite for this dataset.")
    return result


def graph_from_gammagl(graph, spec: DatasetSpec) -> GraphRecord:
    ensure_repo_on_path()
    from gammagl.transforms.graph_serializer import validate_undirected_graph_semantics

    validate_undirected_graph_semantics(graph)
    edge_index = to_list(getattr(graph, "edge_index", [[], []]))
    if len(edge_index) != 2:
        edge_index = [[pair[0] for pair in edge_index], [pair[1] for pair in edge_index]]
    return GraphRecord(
        [[int(item) for item in edge_index[0]], [int(item) for item in edge_index[1]]],
        flatten_feature_ids(getattr(graph, "x", None)),
        flatten_feature_ids(getattr(graph, "edge_attr", None)),
        graph_label(getattr(graph, "y", None), spec.label_width, allow_nan=False),
    )


def validate_splits(size: int, indices: Dict[str, Sequence[int]]) -> None:
    if set(indices) != {"train", "val", "test"}:
        raise ValueError("Dataset must define exactly train, val, and test splits.")
    seen = set()
    for split in ("train", "val", "test"):
        for index in indices[split]:
            if not 0 <= int(index) < size or int(index) in seen:
                raise ValueError("Dataset splits must be in range and disjoint.")
            seen.add(int(index))


def load_dataset_splits(data_root: str, spec: DatasetSpec) -> Dict[str, List[GraphRecord]]:
    ensure_repo_on_path()
    from gammagl import datasets

    try:
        dataset = getattr(datasets, spec.dataset_class)(root=str(data_root))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "GraphTokenizer dataset is not prepared. Run:\n\n"
            "python examples/graph_tokenizer/graph_tokenizer_trainer.py "
            f"--prepare-data --data-root {data_root}\n\n"
            "before starting training.") from error
    indices = {name: [int(item) for item in to_list(values)]
               for name, values in dataset.get_idx_split().items()}
    validate_splits(len(dataset), indices)
    graphs = [graph_from_gammagl(dataset[index], spec) for index in range(len(dataset))]
    return {name: [graphs[index] for index in indices[name]] for name in indices}


def prepare_data(data_root: str) -> None:
    """Materialize all GraphTokenizer datasets from the shared release bundle."""
    ensure_repo_on_path()
    from gammagl.datasets._graph_tokenizer_download import materialize_paper_dataset

    root = Path(data_root)
    print("GraphTokenizer data preparation")
    print(f"Cache: {(root / '.graph_tokenizer_release').resolve()}")
    for spec in DATASETS:
        materialize_paper_dataset(
            dataset_name=spec.storage_name,
            aliases=spec.aliases,
            raw_dir=root / spec.storage_name / "raw",
            cache_root=root,
            allow_download=True,
        )
        print(f"{spec.name}: READY")
    print("Checksum: PASS")
    print(f"Data root: {root.resolve()}")


def synthetic_splits(spec: DatasetSpec) -> Dict[str, List[GraphRecord]]:
    if spec.num_classes:
        labels = [[float(index % spec.num_classes)] for index in range(5)]
    else:
        labels = [[float(index) / 10] for index in range(5)]
    graphs = [
        GraphRecord([[0, 1], [1, 2]], [1, 2, 3], [1, 2], labels[0]),
        GraphRecord([[0, 1], [1, 2]], [2, 3, 4], [1, 3], labels[1]),
        GraphRecord([[0, 1], [1, 2]], [3, 1, 2], [2, 1], labels[2]),
        GraphRecord([[0, 1], [1, 2]], [4, 2, 1], [2, 3], labels[3]),
        GraphRecord([[0, 1], [1, 2]], [1, 4, 2], [3, 1], labels[4]),
    ]
    return {"train": graphs[:3], "val": graphs[3:4], "test": graphs[4:]}


def preprocessing_cache_path(args, spec: DatasetSpec) -> Path:
    import tensorlayerx as tlx

    from gammagl.datasets._graph_tokenizer_download import sha256_file
    from gammagl.transforms.graph_serializer import SERIALIZER_PROTOCOL_VERSION
    from gammagl.transforms.graph_tokenizer import TOKENIZER_SCHEMA_VERSION

    cache_root = Path(getattr(args, "preprocessing_cache_dir", None)
                      or Path(args.data_root) / ".graph_tokenizer_preprocessing_cache")
    raw_dir = Path(args.data_root) / spec.storage_name / "raw"
    processed_dir = Path(args.data_root) / spec.storage_name / "processed"
    processed_files = [processed_dir / f"{tlx.BACKEND}_data_v2.pt",
                       processed_dir / "split_indices_v2.json"]
    if all(path.is_file() for path in processed_files):
        source_kind, paths = "processed", processed_files
    else:
        data_file = next((raw_dir / name for name in ("data.pkl", "data.pkl.gz")
                          if (raw_dir / name).is_file()), None)
        source_kind = "raw"
        paths = ([data_file] if data_file is not None else []) + [
            raw_dir / f"{split}_index.json" for split in ("train", "val", "test")]
    # Dataset loads processed files whenever both exist. Hash the exact inputs
    # it will consume; timestamps can miss in-place replacements of equal size.
    source_files = [(path.name, sha256_file(path)) for path in paths if path.is_file()]
    signature = {
        "version": PREPROCESSING_CACHE_VERSION,
        "dataset": spec.name,
        "dataset_spec": spec.__dict__,
        "encoder": args.encoder,
        "input_configuration": tokenizer_configuration(args),
        "serializer_protocol": SERIALIZER_PROTOCOL_VERSION,
        "tokenizer_schema": TOKENIZER_SCHEMA_VERSION,
        "bpe_merges": args.bpe_merges,
        "bpe_min_frequency": args.bpe_min_frequency,
        "bpe_backend": args.bpe_backend,
        "num_serializations": args.num_serializations,
        "max_length": args.max_length,
        "source_kind": source_kind,
        "source_files": source_files,
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_root / f"{spec.name}-{args.encoder}-{digest}.pickle"


def load_or_prepare_preprocessing(args, spec: DatasetSpec):
    ensure_repo_on_path()
    from gammagl.transforms.graph_tokenizer import TOKENIZER_SCHEMA_VERSION

    cache_path = preprocessing_cache_path(args, spec)
    try:
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
    except FileNotFoundError:
        cached = None
    except (EOFError, ImportError, pickle.UnpicklingError):
        cache_path.unlink(missing_ok=True)
        cached = None
    if cached is not None and (
            not isinstance(cached, dict)
            or cached.get("preprocessing_cache_version") != PREPROCESSING_CACHE_VERSION
            or getattr(cached.get("tokenizer"), "schema_version", None) != TOKENIZER_SCHEMA_VERSION
            or cached.get("input_configuration") != tokenizer_configuration(args)
            or getattr(cached.get("tokenizer"), "fit_scope", None) != tokenizer_configuration(args)["fit_scope"]):
        cache_path.unlink(missing_ok=True)
        cached = None
    if cached is not None:
        try:
            valid = cached.get("tokenizer_fingerprint") == cached["tokenizer"].fingerprint()
        except (AttributeError, RuntimeError):
            valid = False
        if not valid:
            cache_path.unlink(missing_ok=True)
            cached = None
    if cached is not None:
        emit_event("preprocessing_cache", status="hit", path=str(cache_path))
        return (cached["tokenizer"], cached["encoded"], cached["normalizer"],
                cached["output_dim"])

    splits = load_dataset_splits(args.data_root, spec)
    # A first load can generate processed files. Store under their fingerprint
    # so the next call reaches this cache instead of rebuilding indefinitely.
    cache_path = preprocessing_cache_path(args, spec)
    tokenizer = fit_tokenizer(args, splits)
    encoded = encode_splits(tokenizer, splits, args)
    normalizer, output_dim = normalize_labels(encoded, spec)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump({"preprocessing_cache_version": PREPROCESSING_CACHE_VERSION,
                     "input_configuration": tokenizer_configuration(args),
                     "tokenizer_fingerprint": tokenizer.fingerprint(),
                     "tokenizer": tokenizer, "encoded": encoded,
                     "normalizer": normalizer, "output_dim": output_dim}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, cache_path)
    emit_event("preprocessing_cache", status="created", path=str(cache_path))
    return tokenizer, encoded, normalizer, output_dim


def tokenizer_configuration(args):
    dataset = resolve_dataset(getattr(args, "dataset", "aqsol")).name
    preset = PAPER_CONFIGS[(dataset, getattr(args, "encoder", "bert"))]
    fit_scope = getattr(args, "fit_scope", None) or preset["fit_scope"]
    if fit_scope not in ("train", "all"):
        raise ValueError("fit_scope must be train or all.")
    return dict(dataset=dataset, fit_scope=fit_scope,
                serializer=getattr(args, "serializer", None) or preset["serializer"],
                protocol=getattr(args, "token_protocol", None) or "official",
                bpe_encode_mode=getattr(args, "bpe_encode_mode", None) or preset["bpe_encode_mode"])


def fit_tokenizer(args, splits):
    # Official BaseDataLoader.load_data concatenates in this exact order,
    # regardless of raw file order; stable frequency ties depend on it.
    names = ("train", "val", "test") if tokenizer_configuration(args)["fit_scope"] == "all" else ("train",)
    graphs = [graph for name in names for graph in splits[name]]
    ids = [(name, index) for name in names for index in range(len(splits[name]))]
    return make_tokenizer(args, graphs, graph_ids=ids)


def make_tokenizer(args, fit_graphs: Sequence[GraphRecord], graph_ids=None):
    ensure_repo_on_path()
    from gammagl.transforms.graph_bpe import GraphBPE
    from gammagl.transforms.graph_serializer import FrequencyGuidedEulerianSerializer, EulerianSerializer
    from gammagl.transforms.graph_tokenizer import GraphTokenizer, GraphTokenizerSpecialTokens

    special_tokens = GraphTokenizerSpecialTokens()
    if getattr(args, "encoder", "bert") == "gte":
        special_tokens = GraphTokenizerSpecialTokens(pad_token_id=1, unk_token_id=0)

    configuration = tokenizer_configuration(args)
    serializer_cls = {"feuler": FrequencyGuidedEulerianSerializer, "eulerian": EulerianSerializer}[
        configuration["serializer"]]
    tokenizer = GraphTokenizer(
        serializer=serializer_cls(protocol=configuration["protocol"], dataset=configuration["dataset"]),
        bpe=GraphBPE(num_merges=args.bpe_merges, min_frequency=args.bpe_min_frequency,
                     backend=args.bpe_backend),
        special_tokens=special_tokens,
        bpe_encode_mode=configuration["bpe_encode_mode"],
    )
    tokenizer.fit(fit_graphs, graph_ids=(list(range(len(fit_graphs))) if graph_ids is None else graph_ids),
                  num_realizations=args.num_serializations, fit_scope=configuration["fit_scope"])
    return tokenizer


def encode_splits(tokenizer, splits, args) -> Dict[str, List[Dict[str, Any]]]:
    dataset = resolve_dataset(args.dataset).name
    encoded = {}
    for split_name, graphs in splits.items():
        records = []
        for graph_id, graph in enumerate(graphs):
            for start in tokenizer.realization_start_nodes(graph, args.num_serializations):
                result = tokenizer.encode_graph(graph, start_node=start)
                try:
                    tokenizer.validate_token_sequences(
                        [result.input_ids], max_length=args.max_length)
                except ValueError as error:
                    if len(result.input_ids) > args.max_length:
                        raise ValueError(
                            "GraphTokenizer sequence exceeds max_length:\n"
                            f"dataset={dataset}\n"
                            f"split={split_name}\n"
                            f"sample={graph_id}\n"
                            f"length={len(result.input_ids)}\n"
                            f"max_length={args.max_length}\n"
                            "GraphTokenizer does not truncate serialized graphs because "
                            "truncation would discard graph structure.") from error
                    raise
                records.append({
                    "input_ids": result.input_ids,
                    "serialized_token_ids": result.serialized_token_ids,
                    "labels": list(graph.y), "graph_id": graph_id,
                })
        if not records:
            raise ValueError(f"{split_name} split is empty.")
        encoded[split_name] = records
    return encoded


def normalize_labels(encoded, spec: DatasetSpec):
    if spec.num_classes:
        return None, spec.output_dim
    train_values = [[] for _ in range(spec.label_width)]
    for split_name, records in encoded.items():
        for variants in group_records(records):
            label = variants[0]["labels"]
            if not all(math.isfinite(value) for value in label):
                raise ValueError("Normalization requires finite labels in every split.")
            if any(variant["labels"] != label for variant in variants[1:]):
                raise ValueError(
                    f"{split_name} graph_id {variants[0]['graph_id']} has "
                    "inconsistent labels across serializations.")
            if split_name == "train":
                for column, value in enumerate(label):
                    train_values[column].append(value)
    means, stds = [], []
    for values in train_values:
        if not values:
            raise ValueError("Normalization must be fitted from finite training labels only.")
        mean = sum(values) / len(values)
        means.append(mean)
        stds.append(max(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)), 1e-12))
    for records in encoded.values():
        for record in records:
            record["labels"] = [
                (value - means[index]) / stds[index]
                for index, value in enumerate(record["labels"])
            ]
    return {"mean": means, "std": stds, "targets": list(spec.label_names)}, spec.output_dim


def group_records(records: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["graph_id"]), []).append(record)
    return list(grouped.values())


def random_swap_serialized_tokens(tokens, special_ids, probability, ratio, window, rng):
    augmented = list(tokens)
    if rng.random() > float(probability) or len(augmented) <= 1:
        return augmented
    positions = [index for index, token in enumerate(augmented) if int(token) not in special_ids]
    for _ in range(int(len(positions) * float(ratio))):
        center = rng.choice(positions)
        nearby = [index for index in positions
                  if abs(index - center) <= int(window) // 2]
        if len(nearby) >= 2:
            first, second = rng.sample(nearby, 2)
            augmented[first], augmented[second] = augmented[second], augmented[first]
    return augmented


def sequence_mask_serialized_tokens(tokens, special_ids, mask_token_id, probability, ratio, rng):
    augmented = list(tokens)
    if rng.random() > float(probability) or len(augmented) <= 1:
        return augmented
    positions = [index for index, token in enumerate(augmented) if int(token) not in special_ids]
    count = min(len(positions), max(1, min(int(len(positions) * float(ratio)), len(augmented) - 1)))
    for index in rng.sample(positions, count):
        augmented[index] = int(mask_token_id)
    return augmented


def augment_serialized_tokens(tokens, special_ids, augmentation, rng):
    if augmentation is None:
        return list(tokens)
    tokens = random_swap_serialized_tokens(
        tokens, special_ids, augmentation["swap_probability"], augmentation["swap_ratio"],
        augmentation["swap_window"], rng)
    if "mask_probability" in augmentation:
        tokens = sequence_mask_serialized_tokens(
            tokens, special_ids, augmentation["mask_token_id"],
            augmentation["mask_probability"], augmentation["mask_ratio"], rng)
    return tokens


def augment_record_input_ids(
        tokenizer, original_input_ids, serialized_token_ids, special_ids, augmentation,
        max_length, rng):
    if augmentation is None:
        return list(original_input_ids)
    augmented = augment_serialized_tokens(serialized_token_ids, special_ids, augmentation, rng)
    input_ids = tokenizer.encode_tokens(augmented)
    return input_ids if len(input_ids) <= int(max_length) else list(original_input_ids)


def augmentation_config(args, phase):
    prefix = f"{phase}_swap"
    config = {"swap_probability": getattr(args, f"{prefix}_probability"),
              "swap_ratio": getattr(args, f"{prefix}_ratio"),
              "swap_window": getattr(args, f"{prefix}_window")}
    if phase == "finetune":
        config.update({"mask_probability": args.finetune_mask_probability,
                       "mask_ratio": args.finetune_mask_ratio,
                       "mask_token_id": None})
    return config


def make_loader(
        backend, records, batch_size, pad_token_id, shuffle, choose_variant, tokenizer=None,
        augmentation=None, max_length=None, seed=0):
    data = group_records(records) if choose_variant else list(records)
    rng = random.Random(int(seed))

    def collate(items):
        if choose_variant:
            items = [rng.choice(group) for group in items]
        sequences = []
        for item in items:
            if augmentation is None:
                sequences.append(item["input_ids"])
                continue
            if tokenizer is None or max_length is None:
                raise ValueError("Training augmentation requires tokenizer and max_length.")
            config = dict(augmentation)
            config["mask_token_id"] = tokenizer.special_tokens.mask_token_id
            # These are raw loader tokens, whose IDs may overlap model special IDs.
            protected_ids = (set() if tokenizer.serializer.protocol == "official"
                             else mlm_special_token_ids(tokenizer))
            sequences.append(augment_record_input_ids(
                tokenizer, item["input_ids"], item["serialized_token_ids"],
                protected_ids, config, max_length, rng))
        length = max(len(sequence) for sequence in sequences)
        input_ids = np.full((len(items), length), int(pad_token_id), dtype=np.int64)
        attention = np.zeros_like(input_ids)
        for row, tokens in enumerate(sequences):
            input_ids[row, :len(tokens)] = tokens
            attention[row, :len(tokens)] = 1
        return (tlx.convert_to_tensor(input_ids, dtype=tlx.int64),
                tlx.convert_to_tensor(attention, dtype=tlx.int64),
                tlx.convert_to_tensor([item["labels"] for item in items], dtype=tlx.float32),
                tlx.convert_to_tensor([item["graph_id"] for item in items], dtype=tlx.int64))

    return tlx.dataflow.DataLoader(data, batch_size=max(1, int(batch_size)),
                                   shuffle=bool(shuffle), collate_fn=collate)


def set_model_mode(model, training: bool) -> None:
    method = getattr(model, "set_train" if training else "set_eval", None)
    if callable(method):
        method()
    else:
        (model.train if training else model.eval)()


def make_model(args, vocab_size: int, pad_token_id: int, output_dim: int, tokenizer=None):
    ensure_repo_on_path()
    from gammagl.models.graph_bert import GraphBERT
    from gammagl.models.graph_gte import GraphGTE

    kwargs = {"vocab_size": vocab_size, "output_dim": output_dim,
              "pad_token_id": pad_token_id, "task_type": resolve_dataset(args.dataset).task_type,
              "pooling": args.pooling,
              "max_position_embeddings": args.max_position_embeddings}
    for name in ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    if args.encoder == "gte" and not args.allow_random_gte_init:
        if tokenizer is not None:
            if tokenizer.model_id_to_official_id is None:
                raise ValueError("Formal GTE requires an official tokenizer vocabulary mapping.")
            kwargs["model_id_to_official_id"] = tokenizer.model_id_to_official_id
        constructor = GraphGTE.from_official_config if args.gte_reset_weights else GraphGTE.from_pretrained
        model = constructor(cache_dir=args.gte_checkpoint_cache_dir, **kwargs)
        gte_initialization_summary(model, args)
        return model
    return (GraphBERT if args.encoder == "bert" else GraphGTE)(**kwargs)


def gte_initialization_summary(model, args):
    if args.encoder != "gte":
        return None
    if args.allow_random_gte_init:
        return {"pretrained": False, "reproduction": False, "mode": "random_development",
                "initialization_matches_preset": False}
    manifest = getattr(model, "initialization_manifest", None)
    mode = "official_random" if args.gte_reset_weights else "official_pretrained"
    if manifest is None or manifest.get("mode") != mode:
        raise RuntimeError(f"Formal GTE training requires {mode} initialization.")
    if mode == "official_pretrained":
        valid = (manifest.get("pretrained") is True and manifest.get("encoder_coverage") == 1.0
                 and manifest.get("graph_embedding_init") == "official_resize_and_remap"
                 and not manifest.get("missing") and not manifest.get("unexpected")
                 and not manifest.get("shape_mismatches"))
        if not valid:
            raise RuntimeError("Formal GTE training requires complete official checkpoint conversion.")
    return {**manifest, "initialization_matches_preset": True, "reproduction": False}


def mlm_special_token_ids(tokenizer) -> set[int]:
    return {int(token_id) for token_id in vars(tokenizer.special_tokens).values()}


def mlm_random_token_ids(tokenizer) -> List[int]:
    special_ids = mlm_special_token_ids(tokenizer)
    token_ids = sorted({int(token_id) for token_id in tokenizer.vocabulary.values()} - special_ids)
    if not token_ids:
        raise ValueError("MLM random replacement requires a non-special tokenizer vocabulary.")
    return token_ids


def mask_for_mlm(backend, input_ids, attention, tokenizer, probability, generator):
    if backend is tlx:
        ids = tlx.convert_to_numpy(input_ids)
        valid = tlx.convert_to_numpy(attention).astype(bool)
        for token in mlm_special_token_ids(tokenizer):
            valid &= ids != token
        selected = (generator.random(ids.shape) < probability) & valid
        labels, masked = ids.copy(), ids.copy()
        labels[~selected] = -100
        corruption = generator.random(ids.shape)
        masked[selected & (corruption < 0.8)] = int(tokenizer.special_tokens.mask_token_id)
        random_replaced = selected & (corruption >= 0.8) & (corruption < 0.9)
        if random_replaced.any():
            masked[random_replaced] = generator.choice(mlm_random_token_ids(tokenizer), size=int(random_replaced.sum()))
        return (move_tensor(tlx.convert_to_tensor(masked, dtype=tlx.int64), getattr(input_ids, "device", None)),
                move_tensor(tlx.convert_to_tensor(labels, dtype=tlx.int64), getattr(input_ids, "device", None)))
    labels = input_ids.clone()
    blocked = ~attention.bool()
    for token in mlm_special_token_ids(tokenizer):
        blocked |= input_ids.eq(int(token))
    selected = backend.rand(input_ids.shape, generator=generator, device=input_ids.device) < probability
    selected &= ~blocked
    labels[~selected] = -100
    masked = input_ids.clone()
    corruption = backend.rand(input_ids.shape, generator=generator, device=input_ids.device)
    mask_replaced = selected & (corruption < 0.8)
    random_replaced = selected & (corruption >= 0.8) & (corruption < 0.9)
    masked[mask_replaced] = int(tokenizer.special_tokens.mask_token_id)
    if random_replaced.any():
        candidates = backend.as_tensor(mlm_random_token_ids(tokenizer), dtype=backend.long,
                                     device=input_ids.device)
        random_indices = backend.randint(len(candidates), input_ids.shape, generator=generator,
                                       device=input_ids.device)
        masked[random_replaced] = candidates[random_indices[random_replaced]]
    return masked, labels


def mlm_loss(backend, model, input_ids, attention, tokenizer, probability, generator, context="pretrain"):
    """Return the loss sum and valid token count, including for empty batches."""
    masked, labels = mask_for_mlm(backend, input_ids, attention, tokenizer, probability, generator)
    logits = model(masked, attention, task="mlm")
    require_finite_tensor(backend, logits, f"{context} logits")
    valid_indices = np.flatnonzero(tlx.convert_to_numpy(labels).reshape(-1) != -100)
    if not len(valid_indices):
        return tlx.reduce_sum(logits * 0.0), 0
    indices = move_tensor(tlx.convert_to_tensor(valid_indices, dtype=tlx.int64), getattr(logits, "device", None))
    selected_logits = tlx.gather(tlx.reshape(logits, [-1, logits.shape[-1]]), indices)
    selected_labels = tlx.gather(tlx.reshape(labels, [-1]), indices)
    loss = tlx.losses.softmax_cross_entropy_with_logits(selected_logits, selected_labels, reduction="sum")
    return loss, len(valid_indices)


def supervised_logits(backend, model, input_ids, attention, noise_probability, noise_std, generator):
    if noise_probability <= 0.0 or not callable(getattr(model, "_task_logits", None)):
        return model(input_ids, attention, task="supervised")
    pooled = model(input_ids, attention, task="pooled")
    if backend is tlx:
        if generator.random() < float(noise_probability):
            noise = tlx.convert_to_tensor(generator.normal(0.0, float(noise_std), tuple(pooled.shape)), dtype=pooled.dtype)
            pooled = pooled + move_tensor(noise, getattr(pooled, "device", None))
        return model._task_logits(pooled)
    if backend.rand((), generator=generator, device=pooled.device) < float(noise_probability):
        pooled = pooled + backend.randn(
            pooled.shape, generator=generator, dtype=pooled.dtype, device=pooled.device) * float(noise_std)
    return model._task_logits(pooled)


def supervised_loss(backend, logits, labels, spec: DatasetSpec, reduction="mean"):
    if spec.num_classes:
        return tlx.losses.softmax_cross_entropy_with_logits(
            logits, tlx.cast(tlx.reshape(labels, [-1]), tlx.int64), reduction=reduction)
    return tlx.losses.mean_squared_error(logits, labels, reduction=reduction)


def consistency_loss(logits, other_logits, spec: DatasetSpec):
    """Return a target-summed consistency penalty, retaining both gradient paths.

    Classification uses the official symmetric KL (sum over classes and
    samples); regression uses summed squared differences. train_epoch divides
    by the actual target count once, including in accumulation tail windows.
    """
    if spec.num_classes:
        log_p = tlx.logsoftmax(logits, dim=-1)
        log_q = tlx.logsoftmax(other_logits, dim=-1)
        p, q = tlx.softmax(logits, axis=-1), tlx.softmax(other_logits, axis=-1)
        return tlx.reduce_sum((p - q) * (log_p - log_q)) * 0.5
    return tlx.losses.mean_squared_error(logits, other_logits, reduction="sum")


def save_checkpoint(backend, path: Path, model, epoch: int, score: float) -> None:
    require_finite_metrics(score, f"checkpoint epoch={epoch} score")
    ensure_repo_on_path()
    from gammagl.transforms.graph_tokenizer import TOKENIZER_SCHEMA_VERSION

    if tlx.BACKEND == "tensorflow":
        weights = {str(index): weight for index, weight in enumerate(model.all_weights)}
    else:
        weights = model.state_dict()
    for name, weight in weights.items():
        require_finite_tensor(backend, weight, f"checkpoint epoch={epoch} weight {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    state = {"epoch": int(epoch), "score": float(score),
             "artifact_version": ARTIFACT_VERSION,
             "artifact_fingerprint": getattr(model, "artifact_fingerprint", None),
             "tokenizer_schema_version": TOKENIZER_SCHEMA_VERSION,
             "tokenizer_fit_scope": getattr(model, "tokenizer_fit_scope", None),
             "tokenizer_fingerprint": getattr(model, "tokenizer_fingerprint", None)}
    if tlx.BACKEND == "tensorflow":
        arrays = {name: tlx.convert_to_numpy(weight) for name, weight in weights.items()}
        with temporary.open("wb") as handle:
            np.savez(handle, metadata=json.dumps(state, allow_nan=False), **arrays)
    else:
        import torch as torch_backend
        torch_backend.save({"model": weights, **state}, temporary)
    os.replace(temporary, path)


def restore_checkpoint(backend, path: Path, model, device):
    ensure_repo_on_path()
    from gammagl.transforms.graph_tokenizer import TOKENIZER_SCHEMA_VERSION

    if tlx.BACKEND == "tensorflow":
        with np.load(path, allow_pickle=False) as archive:
            state = json.loads(str(archive["metadata"]))
            weights = [archive[str(index)] for index in range(len(archive.files) - 1)]
    else:
        import torch as torch_backend
        try:
            state = torch_backend.load(path, map_location=device, weights_only=False)
        except TypeError:
            state = torch_backend.load(path, map_location=device)
    if state.get("tokenizer_schema_version") != TOKENIZER_SCHEMA_VERSION:
        raise ValueError("Incompatible checkpoint input protocol; retrain with the current tokenizer schema.")
    if ("tokenizer_fit_scope" not in state
            or state["tokenizer_fit_scope"] != getattr(model, "tokenizer_fit_scope", None)):
        raise ValueError("Incompatible checkpoint tokenizer fit scope; retrain with the selected corpus.")
    if ("tokenizer_fingerprint" not in state
            or state["tokenizer_fingerprint"] != getattr(model, "tokenizer_fingerprint", None)):
        raise ValueError("Incompatible checkpoint tokenizer fingerprint; protocol, serializer or vocabulary differ.")
    require_finite_metrics(state["score"], "checkpoint score")
    expected_artifacts = getattr(model, "artifact_fingerprint", None)
    if state.get("artifact_fingerprint") != expected_artifacts or (
            expected_artifacts is not None and state.get("artifact_version") != ARTIFACT_VERSION):
        raise ValueError("Checkpoint artifact fingerprint mismatch; model configuration or normalizer differ.")
    values = enumerate(weights) if tlx.BACKEND == "tensorflow" else state["model"].items()
    for name, weight in values:
        require_finite_tensor(backend, weight, f"checkpoint weight {name}")
    if tlx.BACKEND == "tensorflow":
        if len(weights) != len(model.all_weights) or any(
                tuple(value.shape) != tuple(weight.shape) for value, weight in zip(weights, model.all_weights)):
            raise ValueError("Checkpoint model weights do not match the current model.")
        weights = [tlx.cast(tlx.convert_to_tensor(value), weight.dtype)
                   for value, weight in zip(weights, model.all_weights)]
        for index, weight in enumerate(weights):
            require_finite_tensor(backend, weight, f"checkpoint converted weight {index}")
        tlx.files.assign_weights(weights, model)
    else:
        for name, target in model.state_dict().items():
            if name in state["model"]:
                converted = state["model"][name].to(dtype=target.dtype)
                require_finite_tensor(backend, converted, f"checkpoint converted weight {name}")
        model.load_state_dict(state["model"])
    return state


def gradient_accumulation_steps(args) -> int:
    steps = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    if steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive.")
    return steps


def optimizer_steps_per_epoch(loader, args) -> int:
    return math.ceil(len(loader) / gradient_accumulation_steps(args))


def step_accumulated_gradients(backend, model, optimizer, scheduler, target_count, max_grad_norm,
                              context="training"):
    # Normalize by targets before clipping; an empty MLM window makes no update
    # (including weight decay) and must not advance the learning-rate schedule.
    if target_count:
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                parameter.grad.div_(target_count)
                require_finite_tensor(backend, parameter.grad, f"{context} gradient {name}")
        norm = backend.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        # The total norm can overflow even when every gradient element is finite.
        if norm is not None:
            require_finite_tensor(backend, norm, f"{context} gradient norm")
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                require_finite_tensor(backend, parameter.grad, f"{context} clipped gradient {name}")
            require_finite_tensor(backend, parameter, f"{context} weight {name}")
        optimizer.step()
        for name, parameter in model.named_parameters():
            require_finite_tensor(backend, parameter, f"{context} updated weight {name}")
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def train_epoch(backend, model, loader, optimizer, scheduler, args, device, seed, epoch,
                phase, tokenizer=None, spec=None):
    set_model_mode(model, True)
    helpers = training_helpers()
    native = hasattr(optimizer, "apply_gradients")
    accumulator = helpers.GradientAccumulator(model, optimizer, require_finite_tensor) if native else None
    generator = (np.random.default_rng(seed) if backend is tlx else
                 backend.Generator(device=device).manual_seed(seed))
    total, count, window_count = 0.0, 0, 0
    accumulation_steps = gradient_accumulation_steps(args)
    max_norm = args.pretrain_max_grad_norm if phase == "pretrain" else args.finetune_max_grad_norm
    if not native:
        optimizer.zero_grad(set_to_none=True)

    def compute(data, labels):
        input_ids, attention = data
        if phase == "pretrain":
            loss, target_count = mlm_loss(backend, model, input_ids, attention, tokenizer,
                                         args.mask_probability, generator, context)
        else:
            logits = supervised_logits(backend, model, input_ids, attention,
                args.finetune_noise_probability, args.finetune_noise_std, generator)
            require_finite_tensor(backend, logits, f"{context} logits")
            loss = supervised_loss(backend, logits, labels, spec, reduction="sum")
            require_finite_tensor(backend, loss, f"{context} task loss")
            if getattr(args, "finetune_use_consistency_regularization", False):
                other_logits = supervised_logits(backend, model, input_ids, attention,
                    args.finetune_noise_probability, args.finetune_noise_std, generator)
                require_finite_tensor(backend, other_logits, f"{context} second logits")
                other_loss = supervised_loss(backend, other_logits, labels, spec, reduction="sum")
                require_finite_tensor(backend, other_loss, f"{context} second task loss")
                penalty = consistency_loss(logits, other_logits, spec)
                require_finite_tensor(backend, penalty, f"{context} consistency loss")
                loss = (loss + other_loss) * 0.5 + getattr(args, "finetune_consistency_alpha", 1.0) * penalty
            target_count = (math.prod(labels.shape) if spec.num_classes else
                            math.prod(np.broadcast_shapes(tuple(logits.shape), tuple(labels.shape))))
        require_finite_tensor(backend, loss, f"{context} loss")
        return loss, target_count

    loss_network = helpers.GraphTokenLoss(model, compute)
    for batch_index, (input_ids, attention, labels, _) in enumerate(loader):
        context = f"{phase} epoch={epoch} batch={batch_index + 1}"
        input_ids, attention, labels = (move_tensor(value, device) for value in (input_ids, attention, labels))
        with helpers.gradient_context() as tape:
            loss = loss_network((input_ids, attention), labels)
        target_count = loss_network.target_count
        if target_count:
            if native:
                accumulator.collect(loss, tape)
            else:
                # Accept raw torch optimizers for callers comparing native autograd
                # against the TLX accumulation path; normal runs use TLX optimizers.
                loss.backward()
        window_count += target_count
        if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader):
            if native:
                accumulator.step(window_count, max_norm, scheduler, context)
            else:
                step_accumulated_gradients(backend, model, optimizer, scheduler, window_count, max_norm, context)
            window_count = 0
        total += float(tlx.convert_to_numpy(loss))
        count += target_count
    train_loss = total / max(count, 1)
    require_finite_metrics(train_loss, f"{phase} epoch={epoch} train_loss")
    return train_loss


def train_mlm_epoch(backend, model, loader, optimizer, scheduler, tokenizer, args, device, seed, epoch=None):
    return train_epoch(backend, model, loader, optimizer, scheduler, args, device, seed, epoch,
                       "pretrain", tokenizer=tokenizer)


def run_mlm(backend, model, train_loader, tokenizer, args, device):
    optimizer = make_optimizer(args.pretrain_learning_rate,
        getattr(args, "pretrain_weight_decay", getattr(args, "weight_decay", 0.0)))
    scheduler = build_warmup_cosine_scheduler(
        backend, optimizer, total_steps=optimizer_steps_per_epoch(train_loader, args) * args.pretrain_epochs,
        warmup_ratio=args.pretrain_warmup_ratio)
    history = []
    for epoch in range(1, args.pretrain_epochs + 1):
        train_loss = train_mlm_epoch(backend, model, train_loader, optimizer, scheduler, tokenizer,
                                     args, device, args.seed + epoch, epoch)
        require_finite_metrics(train_loss, f"pretrain epoch={epoch} train_loss")
        history.append({"epoch": epoch, "train_loss": train_loss})
        emit_event("epoch", phase="pretrain", epoch=epoch, train_loss=train_loss)
    return {"epochs_completed": len(history), "final_loss": history[-1]["train_loss"],
            "history": history}


def denormalize(backend, values, normalizer):
    if normalizer is None:
        return values
    mean = move_tensor(tlx.convert_to_tensor(normalizer["mean"], dtype=values.dtype), getattr(values, "device", None))
    std = move_tensor(tlx.convert_to_tensor(normalizer["std"], dtype=values.dtype), getattr(values, "device", None))
    return values * std + mean


def evaluate_downstream(backend, model, loader, spec, normalizer, device, phase="evaluation", epoch=None):
    set_model_mode(model, False)
    total_loss, count, grouped = 0.0, 0, {}
    if tlx.BACKEND == "torch":
        import torch as torch_backend
        evaluation_context = torch_backend.no_grad()
    else:
        evaluation_context = nullcontext()
    with evaluation_context:
        for batch_index, (input_ids, attention, labels, graph_ids) in enumerate(loader):
            context = f"{phase} epoch={epoch} batch={batch_index + 1}"
            input_ids, attention, labels = (move_tensor(value, device) for value in (input_ids, attention, labels))
            logits = model(input_ids, attention, task="supervised")
            require_finite_tensor(backend, logits, f"{context} logits")
            loss = supervised_loss(backend, logits, labels, spec)
            require_finite_tensor(backend, loss, f"{context} loss")
            total_loss += float(tlx.convert_to_numpy(loss)) * len(input_ids)
            count += len(input_ids)
            predictions = tlx.softmax(logits, axis=-1) if spec.num_classes else logits
            for graph_id, label, prediction in zip(tlx.convert_to_numpy(graph_ids).tolist(),
                    tlx.convert_to_numpy(labels), tlx.convert_to_numpy(predictions)):
                grouped.setdefault(graph_id, {"label": label, "predictions": []})["predictions"].append(prediction)
    labels = np.stack([item["label"] for item in grouped.values()])
    predictions = np.stack([np.stack(item["predictions"]).mean(0) for item in grouped.values()])
    if not np.isfinite(predictions).all():
        raise FloatingPointError(f"Non-finite {phase} epoch={epoch} grouped logits")
    if spec.num_classes:
        metric = float((predictions.argmax(axis=-1) == labels.reshape(-1).astype(int)).mean())
        detail = {spec.metric.lower(): metric, "metric_space": "class_label"}
    else:
        if normalizer is not None:
            mean, std = np.asarray(normalizer["mean"]), np.asarray(normalizer["std"])
            labels, predictions = labels * std + mean, predictions * std + mean
        metric = float(np.abs(labels - predictions).mean())
        detail = {spec.metric.lower(): metric, "metric_space": "raw_label"}
    result = {"loss": total_loss / max(count, 1), "metric": metric,
              "num_graphs": len(grouped), **detail}
    require_finite_metrics(result, f"{phase} epoch={epoch}")
    return result


def train_downstream_epoch(backend, model, loader, optimizer, scheduler, spec, args, device, seed=0, epoch=None):
    return train_epoch(backend, model, loader, optimizer, scheduler, args, device, seed, epoch,
                       "finetune", spec=spec)


def run_finetuning(backend, model, train_loader, val_loader, test_loader, spec, normalizer, args, device, checkpoint):
    optimizer = make_optimizer(args.learning_rate,
        getattr(args, "finetune_weight_decay", getattr(args, "weight_decay", 0.0)))
    scheduler = build_warmup_cosine_scheduler(
        backend, optimizer, total_steps=optimizer_steps_per_epoch(train_loader, args) * args.finetune_epochs,
        warmup_ratio=args.finetune_warmup_ratio)
    best, stale, history = None, 0, []
    higher_is_better = spec.higher_is_better
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss = train_downstream_epoch(
            backend, model, train_loader, optimizer, scheduler, spec, args, device, args.seed + epoch, epoch)
        require_finite_metrics(train_loss, f"finetune epoch={epoch} train_loss")
        validation = evaluate_downstream(backend, model, val_loader, spec, normalizer, device, "validation", epoch)
        require_finite_metrics(validation, f"validation epoch={epoch}")
        score = validation["metric"]
        history.append({"epoch": epoch, "train_loss": train_loss, "validation": validation})
        emit_event("epoch", phase="finetune", epoch=epoch, train_loss=train_loss,
                   validation_metric=score)
        improved = best is None or (score > best if higher_is_better else score < best)
        if improved:
            best, stale = score, 0
            save_checkpoint(backend, checkpoint, model, epoch, score)
        else:
            stale += 1
            if stale >= args.early_stopping_patience:
                break
    state = restore_checkpoint(backend, checkpoint, model, device)
    # This is the only downstream test evaluation.
    test = evaluate_downstream(backend, model, test_loader, spec, normalizer, device, "test", state["epoch"])
    require_finite_metrics(test, f"test epoch={state['epoch']}")
    return {"best_epoch": state["epoch"], "best_validation_metric": state["score"],
            "use_consistency_regularization": bool(getattr(args, "finetune_use_consistency_regularization", False)),
            "consistency_alpha": float(getattr(args, "finetune_consistency_alpha", 1.0)),
            "history": history, "test": test}


def apply_preset(args, config=None):
    if config is None:
        config = PAPER_CONFIGS[(resolve_dataset(args.dataset).name, args.encoder)]
    pretrain_batch_size = args.pretrain_batch_size
    finetune_batch_size = args.finetune_batch_size
    for name, value in config.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    if args.batch_size is not None:
        if pretrain_batch_size is None:
            args.pretrain_batch_size = args.batch_size
        if finetune_batch_size is None:
            args.finetune_batch_size = args.batch_size
    if args.max_length > args.max_position_embeddings:
        raise ValueError(
            "max_length must not exceed max_position_embeddings.")
    gradient_accumulation_steps(args)
    if not math.isfinite(args.finetune_consistency_alpha) or args.finetune_consistency_alpha < 0.0:
        raise ValueError("finetune consistency alpha must be finite and non-negative.")
    return args


def metric_name_for_dataset(spec):
    return spec.metric


def experiment_output_directory(args, spec):
    directory = Path(args.output_dir) / spec.name / args.encoder
    return directory / "train_only" if tokenizer_configuration(args)["fit_scope"] == "train" else directory


def run_output_directory(args, spec, run_index, seed):
    return experiment_output_directory(args, spec) / f"run_{run_index + 1:02d}_seed_{seed}"


def run_single_experiment(args, config, seed, run_index=0):
    args = apply_preset(argparse.Namespace(**vars(args)), config)
    args.seed = int(seed)
    backend = ensure_tlx_backend()
    runtime = configure_device(tlx.BACKEND, args.device)
    emit_event("runtime", **runtime)
    set_seed(backend, args.seed)
    spec = resolve_dataset(args.dataset)
    emit_event("stage", stage="tokenizer_fit", status="started")
    if args.smoke:
        splits = synthetic_splits(spec)
        tokenizer = fit_tokenizer(args, splits)
        encoded = encode_splits(tokenizer, splits, args)
        normalizer, output_dim = normalize_labels(encoded, spec)
    else:
        tokenizer, encoded, normalizer, output_dim = load_or_prepare_preprocessing(args, spec)
    emit_event("stage", stage="tokenizer_fit", status="completed")
    emit_event("stage", stage="encoding", status="started")
    emit_event("stage", stage="encoding", status="completed")
    tokenizer.validate_model_vocab(tokenizer.max_token_id + 1)
    device = args.device
    model = make_model(args, tokenizer.max_token_id + 1, tokenizer.special_tokens.pad_token_id, output_dim, tokenizer)
    model.tokenizer_fingerprint = tokenizer.fingerprint()
    model.tokenizer_fit_scope = tokenizer.fit_scope
    if tlx.BACKEND == "torch":
        model = model.to(device)
    loader_args = {"backend": backend, "pad_token_id": tokenizer.special_tokens.pad_token_id}
    pretrain_loader = make_loader(
        **loader_args, batch_size=args.pretrain_batch_size, records=encoded["train"], shuffle=True, choose_variant=True,
        tokenizer=tokenizer, augmentation=augmentation_config(args, "pretrain"),
        max_length=args.max_length, seed=args.seed)
    train_loader = make_loader(
        **loader_args, batch_size=args.finetune_batch_size, records=encoded["train"], shuffle=True, choose_variant=True,
        tokenizer=tokenizer, augmentation=augmentation_config(args, "finetune"),
        max_length=args.max_length, seed=args.seed + 1)
    val_loader = make_loader(**loader_args, batch_size=args.finetune_batch_size, records=encoded["val"],
                             shuffle=False, choose_variant=False)
    test_loader = make_loader(**loader_args, batch_size=args.finetune_batch_size, records=encoded["test"],
                              shuffle=False, choose_variant=False)
    directory = run_output_directory(args, spec, run_index, args.seed)
    checkpoint = directory / ("best_finetune.npz" if tlx.BACKEND == "tensorflow" else "best_finetune.pt")
    from examples.graph_tokenizer import artifacts
    artifacts.save_run(directory, args, spec, model, tokenizer, normalizer, encoded, runtime,
                       synthetic=splits if args.smoke else None)
    mlm = run_mlm(backend, model, pretrain_loader, tokenizer, args, device)
    finetune = run_finetuning(backend, model, train_loader, val_loader, test_loader, spec, normalizer,
                              args, device, checkpoint)
    summary = {"dataset": spec.name, "encoder": args.encoder, "seed": args.seed,
            "run_index": run_index, "metric_name": metric_name_for_dataset(spec),
            "metric": finetune["test"]["metric"],
            "tokenizer_fit_split": tokenizer.fit_scope, "tokenizer_fit_scope": tokenizer.fit_scope,
            "tokenizer_fingerprint": tokenizer.fingerprint(),
            "normalization_fit_split": "train",
            "artifacts": artifacts.completed_run(directory, checkpoint),
            "mlm": mlm, "finetune": finetune,
            "checkpoints": {"finetune": str(checkpoint)}}
    if args.encoder == "gte":
        summary["gte_initialization"] = gte_initialization_summary(model, args)
    return summary


def aggregate_run_results(results):
    if not results:
        raise ValueError("Cannot aggregate zero GraphTokenizer runs")
    for result in results:
        validate_run_result(result)
    scopes = {result.get("tokenizer_fit_scope") for result in results}
    if len(scopes) != 1:
        raise ValueError("Cannot aggregate runs with different tokenizer fit scopes.")
    metrics = [result["metric"] for result in results]
    mean = statistics.mean(metrics)
    try:
        std = statistics.stdev(metrics) if len(metrics) > 1 else 0.0
    except OverflowError as error:
        raise FloatingPointError("Non-finite aggregate std: overflow") from error
    require_finite_metrics({"mean": mean, "std": std}, "aggregate")
    first = results[0]
    return {"dataset": first["dataset"], "encoder": first["encoder"],
            "tokenizer_fit_scope": first.get("tokenizer_fit_scope"),
            "seeds": [result["seed"] for result in results], "num_runs": len(results),
            "metric_name": first["metric_name"], "run_metrics": metrics,
            "mean": mean, "std": std, "std_ddof": 1,
            "display": f"{mean} ± {std}", "complete": True}


def build_parser():
    parser = argparse.ArgumentParser(description="Train GraphTokenizer with validation-selected checkpoints.")
    parser.add_argument("--dataset", default="mutagenicity")
    parser.add_argument("--encoder", "--model", dest="encoder", choices=("bert", "gte"), default="bert")
    parser.add_argument("--fit-scope", choices=("train", "all"),
                        help="Tokenizer fitting corpus: all (official preset) or train (inductive variant).")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="runs/graph_tokenizer")
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--pretrain-batch-size", type=int)
    parser.add_argument("--finetune-batch-size", type=int)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float)
    parser.add_argument("--pretrain-learning-rate", "--pretrain-lr", dest="pretrain_learning_rate", type=float)
    parser.add_argument("--pretrain-epochs", "--pretrain-epoch", dest="pretrain_epochs", type=int)
    parser.add_argument("--finetune-epochs", "--n-epoch", dest="finetune_epochs", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-position-embeddings", type=int)
    parser.add_argument("--bpe-merges", "--num-merges", dest="bpe_merges", type=int)
    parser.add_argument("--bpe-min-frequency", "--min-frequency", dest="bpe_min_frequency", type=int)
    parser.add_argument("--bpe-backend", choices=("python", "auto", "cpp"), default="python")
    parser.add_argument("--serializer", choices=("feuler", "eulerian"))
    parser.add_argument("--token-protocol", choices=("official", "local_refs"))
    parser.add_argument("--bpe-encode-mode", choices=("all", "none"))
    parser.add_argument("--preprocessing-cache-dir")
    parser.add_argument("--num-serializations", "--num-realizations", dest="num_serializations", type=int)
    parser.add_argument("--early-stopping-patience", "--patience", dest="early_stopping_patience", type=int)
    parser.add_argument("--mask-probability", "--mask-prob", dest="mask_probability", type=float)
    parser.add_argument("--pretrain-weight-decay", type=float)
    parser.add_argument("--finetune-weight-decay", type=float)
    parser.add_argument("--pretrain-warmup-ratio", type=float)
    parser.add_argument("--finetune-warmup-ratio", type=float)
    parser.add_argument("--pretrain-max-grad-norm", type=float)
    parser.add_argument("--finetune-max-grad-norm", type=float)
    parser.add_argument("--pretrain-swap-probability", type=float)
    parser.add_argument("--pretrain-swap-ratio", type=float)
    parser.add_argument("--pretrain-swap-window", type=int)
    parser.add_argument("--finetune-swap-probability", type=float)
    parser.add_argument("--finetune-swap-ratio", type=float)
    parser.add_argument("--finetune-swap-window", type=int)
    parser.add_argument("--finetune-mask-probability", type=float)
    parser.add_argument("--finetune-mask-ratio", type=float)
    parser.add_argument("--finetune-noise-probability", type=float)
    parser.add_argument("--finetune-noise-std", type=float)
    parser.add_argument("--finetune-use-consistency-regularization", action=argparse.BooleanOptionalAction,
                        help="Use two stochastic supervised passes and the official consistency loss.")
    parser.add_argument("--finetune-consistency-alpha", type=float,
                        help="Non-negative consistency loss weight (official default: 1.0).")
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--pooling", choices=("mean", "cls"))
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument("--seed", type=int,
                              help="Compatibility alias for one explicit seed.")
    seed_options.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
                              help="Independent training seeds (default: 42 43 44 45 46).")
    parser.add_argument("--gte-checkpoint-cache-dir")
    parser.add_argument("--prepare-data", action="store_true",
                        help="Download and materialize shared GraphTokenizer data, then exit.")
    parser.add_argument("--allow-random-gte-init", action="store_true",
                        help="Development only; this is not a paper reproduction run.")
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-hidden-layers", type=int)
    parser.add_argument("--num-attention-heads", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--smoke", action="store_true", help="Use five tiny synthetic graphs and small model defaults.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args._cli_args = list(sys.argv[1:] if argv is None else argv)
    if args.seed is not None:
        args.seeds = [args.seed]
    if args.prepare_data:
        prepare_data(args.data_root)
        return None
    if args.smoke and args.encoder == "gte" and not args.allow_random_gte_init:
        raise ValueError(
            "GTE tiny smoke architecture is incompatible with the pretrained GTE checkpoint. "
            "For a development-only tiny pipeline smoke, explicitly pass "
            "--allow-random-gte-init; it is not a reproduced result.")
    if args.smoke:
        smoke_overrides = {"batch_size": 2, "pretrain_batch_size": 2,
                           "finetune_batch_size": 2, "pretrain_epochs": 2, "finetune_epochs": 2,
                           "max_length": 32, "max_position_embeddings": 32,
                           "bpe_merges": 8, "num_serializations": 1,
                           "early_stopping_patience": 1,
                           "hidden_size": 16, "num_hidden_layers": 1,
                           "num_attention_heads": 4, "intermediate_size": 32}
        for name, value in smoke_overrides.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
    spec = resolve_dataset(args.dataset)
    config = PAPER_CONFIGS[(spec.name, args.encoder)]
    results = []
    summary_path = experiment_output_directory(args, spec) / "summary.json"
    try:
        for run_index, seed in enumerate(args.seeds):
            print(f"=== Run {run_index + 1}/{len(args.seeds)} | seed={seed} ===", flush=True)
            result = run_single_experiment(args, config, seed, run_index=run_index)
            validate_run_result(result)
            serialized_result = json.dumps(result, indent=2, allow_nan=False)
            directory = run_output_directory(args, spec, run_index, seed)
            directory.mkdir(parents=True, exist_ok=True)
            atomic_write(directory / "summary.json", serialized_result.encode("utf-8"))
            results.append(result)
            print(f"=== Run {run_index + 1}/{len(args.seeds)} completed ===\n"
                  f"{result['metric_name']}: {result['metric']}", flush=True)
        summary = aggregate_run_results(results)
        serialized = json.dumps(summary, indent=2, allow_nan=False)
    except Exception:
        partial = {"dataset": spec.name, "encoder": args.encoder,
                   "tokenizer_fit_scope": tokenizer_configuration(args)["fit_scope"],
                   "seeds": list(args.seeds), "num_runs": len(args.seeds),
                   "metric_name": metric_name_for_dataset(spec),
                   "run_metrics": [result["metric"] for result in results],
                   "complete": False}
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        # Only fully validated successful runs enter results. Do not serialize
        # the rejected result (which may itself contain NaN/Inf) on this path.
        atomic_write(summary_path, json.dumps(partial, indent=2, allow_nan=False).encode("utf-8"))
        raise
    atomic_write(summary_path, serialized.encode("utf-8"))
    print(serialized, flush=True)
    return summary


if __name__ == "__main__":
    main()
