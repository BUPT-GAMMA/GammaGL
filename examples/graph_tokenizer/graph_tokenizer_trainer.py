import argparse
import copy
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import pickle
import platform
import random
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    canonical_name: str
    aliases: Tuple[str, ...]
    task_type: str
    output_dim: int
    metric: str
    paper_name: str
    data_dir_names: Tuple[str, ...] = ()
    label_keys: Tuple[str, ...] = ()


SUPPORTED_MODELS = ("bert", "gte")


DATASET_SPECS = (
    DatasetSpec(
        canonical_name="qm9",
        aliases=("qm9",),
        task_type="regression",
        output_dim=16,
        metric="mae",
        paper_name="QM9",
        data_dir_names=("qm9",),
        label_keys=(
            "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0",
            "u298", "h298", "g298", "cv", "u0_atom", "u298_atom",
            "h298_atom", "g298_atom",
        ),
    ),
    DatasetSpec(
        canonical_name="molhiv",
        aliases=("molhiv", "ogbg-molhiv", "ogbg_molhiv"),
        task_type="binary_classification",
        output_dim=1,
        metric="rocauc",
        paper_name="OGBG-molhiv",
        data_dir_names=("molhiv", "ogbg-molhiv", "ogbg_molhiv"),
        label_keys=("label",),
    ),
    DatasetSpec(
        canonical_name="peptides-func",
        aliases=("peptides-func", "peptides_func", "p-func", "p_func"),
        task_type="multi_label_classification",
        output_dim=10,
        metric="ap",
        paper_name="Peptides-func",
        data_dir_names=("peptides-func", "peptides_func", "p-func", "p_func"),
        label_keys=("labels",),
    ),
    DatasetSpec(
        canonical_name="peptides-struct",
        aliases=("peptides-struct", "peptides_struct", "p-struct", "p_struct"),
        task_type="multi_target_regression",
        output_dim=11,
        metric="average_mae",
        paper_name="Peptides-struct",
        data_dir_names=("peptides-struct", "peptides_struct", "p-struct", "p_struct"),
        label_keys=("labels",),
    ),
)


class SyntheticGraph:
    def __init__(self, edge_index, x, edge_attr=None, y=None):
        self.edge_index = edge_index
        self.x = x
        self.edge_attr = edge_attr
        self.y = y
        self.num_nodes = len(x)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_repo_on_path() -> Path:
    root = repo_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def resolve_dataset_spec(name: str) -> DatasetSpec:
    normalized = name.lower().replace("_", "-")
    for spec in DATASET_SPECS:
        if normalized in {alias.replace("_", "-") for alias in spec.aliases}:
            return spec
    supported = ", ".join(spec.paper_name for spec in DATASET_SPECS)
    raise ValueError(f"Unsupported dataset '{name}'. Supported datasets: {supported}.")


def resolve_model_name(name: str) -> str:
    normalized = str(name).lower()
    if normalized not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unsupported model '{name}'. Supported models: {supported}.")
    return normalized


def effective_seed(seed: Optional[int]) -> int:
    return 0 if seed is None else int(seed)


def make_synthetic_graphs(spec: DatasetSpec) -> List[SyntheticGraph]:
    targets = synthetic_targets(spec)
    return [
        SyntheticGraph(edge_index=[[0, 1], [1, 2]], x=[1, 2, 3], edge_attr=[1, 2], y=targets[0]),
        SyntheticGraph(edge_index=[[0, 0], [1, 2]], x=[1, 2, 4], edge_attr=[1, 3], y=targets[1]),
        SyntheticGraph(edge_index=[[0, 1], [1, 0]], x=[2, 1], edge_attr=[1, 1], y=targets[2]),
    ]


def synthetic_targets(spec: DatasetSpec) -> List[List[float]]:
    if spec.metric == "rocauc":
        return [[0.0], [1.0], [0.0]]
    if spec.metric == "ap":
        rows = []
        for index in range(3):
            row = [0.0] * spec.output_dim
            row[index % spec.output_dim] = 1.0
            rows.append(row)
        return rows
    return [[float(i + j) / 10.0 for j in range(spec.output_dim)] for i in range(3)]


def load_benchmark_splits(data_root, spec: DatasetSpec) -> Dict[str, List[SyntheticGraph]]:
    dataset = load_gammagl_benchmark_dataset(data_root, spec)
    graphs = [gammagl_graph_to_synthetic_graph(dataset[index], spec) for index in range(len(dataset))]
    return select_strict_splits(graphs, dataset.get_idx_split())


def load_gammagl_benchmark_dataset(data_root, spec: DatasetSpec):
    ensure_repo_on_path()
    dataset_cls = {
        "qm9": "QM9",
        "molhiv": "OGBGMolHIV",
        "peptides-struct": "PeptidesStruct",
    }.get(spec.canonical_name)
    if dataset_cls is None:
        raise ValueError(
            "GammaGL GraphTokenizer benchmarks support QM9, OGBG-molhiv, "
            "and Peptides-struct only.")

    from gammagl import datasets as gammagl_datasets
    return getattr(gammagl_datasets, dataset_cls)(root=str(data_root))


def validate_split_indices(num_samples: int, split_indices: Dict[str, List[int]]) -> None:
    required = ("train", "val", "test")
    if set(split_indices) != set(required):
        raise ValueError(f"Split files must define exactly {required}.")
    seen = {}
    for split_name in required:
        for index in split_indices[split_name]:
            if index < 0 or index >= num_samples:
                raise ValueError(
                    f"{split_name} split index {index} is outside [0, {num_samples}).")
            if index in seen:
                raise ValueError(
                    f"{split_name} split index {index} overlaps {seen[index]} split.")
            seen[index] = split_name


def select_strict_splits(graphs: Sequence[SyntheticGraph], split_indices: Dict[str, List[int]]):
    validate_split_indices(len(graphs), split_indices)
    splits = {}
    for split_name in ("train", "val", "test"):
        split_graphs = [graphs[index] for index in split_indices[split_name]]
        for graph, graph_id in zip(split_graphs, split_indices[split_name]):
            graph.graph_tokenizer_id = int(graph_id)
        splits[split_name] = split_graphs
    return splits


def gammagl_graph_to_synthetic_graph(graph, spec: DatasetSpec) -> SyntheticGraph:
    return SyntheticGraph(
        edge_index=normalize_edge_index(getattr(graph, "edge_index", None)),
        x=flatten_feature_ids(getattr(graph, "x", None)),
        edge_attr=flatten_feature_ids(getattr(graph, "edge_attr", None)),
        y=as_float_list(
            getattr(graph, "y", None),
            spec.output_dim,
            allow_nan=spec.canonical_name == "peptides-struct",
        ),
    )


def normalize_edge_index(edge_index) -> List[List[int]]:
    if edge_index is None:
        return [[], []]
    values = to_list(edge_index)
    if len(values) == 2 and is_sequence(values[0]) and is_sequence(values[1]):
        return [[int(value) for value in values[0]], [int(value) for value in values[1]]]
    src = [int(pair[0]) for pair in values]
    dst = [int(pair[1]) for pair in values]
    return [src, dst]


def flatten_feature_ids(values) -> List[int]:
    if values is None:
        return []
    values = to_list(values)
    flattened = []
    for item in values:
        if is_sequence(item):
            if len(item) != 1 or is_sequence(item[0]):
                raise ValueError(
                    "Multi-dimensional features require a dataset-specific token adapter.")
            item = item[0]
        flattened.append(int(item))
    return flattened


def as_float_list(value, output_dim: int, allow_nan: bool = False) -> List[float]:
    if value is None:
        raise ValueError("Sample is missing its label.")
    values = to_list(value)
    while is_sequence(values) and len(values) == 1 and is_sequence(values[0]):
        values = values[0]
    if not is_sequence(values):
        values = [values]
    if len(values) != output_dim:
        raise ValueError(
            f"Label must have exactly {output_dim} values; received {len(values)}.")
    result = [float(item) for item in values]
    if not allow_nan and any(math.isnan(item) for item in result):
        raise ValueError("Labels cannot contain NaN values for this dataset.")
    return result


def to_list(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def is_sequence(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict))


def compute_primary_metric(spec: DatasetSpec, y_true, y_pred) -> float:
    if spec.metric == "average_mae":
        return average_mae(y_true, y_pred)
    if spec.metric == "mae":
        return mean_absolute_error(y_true, y_pred)
    if spec.metric == "rocauc":
        return binary_roc_auc(flatten_numeric(y_true), flatten_numeric(y_pred))
    if spec.metric == "ap":
        true_rows = as_2d_float_rows(y_true)
        pred_rows = as_2d_float_rows(y_pred)
        if not true_rows:
            return float("nan")
        num_tasks = max(len(row) for row in true_rows)
        scores = []
        for task_index in range(num_tasks):
            task_true = [row[task_index] for row in true_rows if task_index < len(row)]
            task_pred = [row[task_index] for row in pred_rows if task_index < len(row)]
            if len(set(task_true)) < 2:
                continue
            scores.append(average_precision(task_true, task_pred))
        return sum(scores) / len(scores) if scores else float("nan")
    if spec.metric in {"acc", "accuracy"}:
        return accuracy_score(y_true, y_pred)
    raise ValueError(f"Unsupported metric: {spec.metric}")


def compute_task_metrics(spec: DatasetSpec, y_true, y_pred) -> Dict[str, float]:
    primary = compute_primary_metric(spec, y_true, y_pred)
    if spec.metric == "rocauc":
        return {"auc": primary}
    if spec.metric == "ap":
        return {"ap": primary}
    if spec.metric == "average_mae":
        return {"average_mae": primary, "mae": mean_absolute_error(y_true, y_pred)}
    if spec.metric == "mae":
        return {"mae": primary}
    if spec.metric in {"acc", "accuracy"}:
        return {"acc": primary}
    raise ValueError(f"Unsupported metric: {spec.metric}")


def primary_metric_name(spec: DatasetSpec) -> str:
    if spec.metric == "rocauc":
        return "auc"
    if spec.metric == "accuracy":
        return "acc"
    return spec.metric


def higher_is_better(spec: DatasetSpec) -> bool:
    return spec.metric in {"rocauc", "ap", "acc", "accuracy"}


def metric_is_better(spec: DatasetSpec, value: float, best_value: Optional[float]) -> bool:
    if math.isnan(value):
        return False
    if best_value is None or math.isnan(best_value):
        return True
    return value > best_value if higher_is_better(spec) else value < best_value


def select_best_epoch(spec: DatasetSpec, history: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    metric_name = primary_metric_name(spec)
    best_entry = None
    best_value = None
    for entry in history:
        value = entry.get("val", {}).get("metrics", {}).get(metric_name, float("nan"))
        if metric_is_better(spec, float(value), best_value):
            best_entry = entry
            best_value = float(value)
    return best_entry


def mean(values: Sequence[float]) -> float:
    values = [float(value) for value in values if not math.isnan(float(value))]
    return sum(values) / len(values) if values else float("nan")


def population_std(values: Sequence[float]) -> float:
    values = [float(value) for value in values if not math.isnan(float(value))]
    if not values:
        return float("nan")
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def format_result_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    metric_name = summary["primary_metric"]
    return {
        "dataset": summary["dataset"],
        "model": summary["model"],
        "run": summary.get("run", 0),
        "seed": summary.get("seed"),
        "best_epoch": summary.get("best_epoch"),
        f"val_{metric_name}": summary.get("best_val", {}).get("metrics", {}).get(metric_name, float("nan")),
        f"test_{metric_name}": summary.get("best_test", {}).get("metrics", {}).get(metric_name, float("nan")),
        "val_loss": summary.get("best_val", {}).get("loss", float("nan")),
        "test_loss": summary.get("best_test", {}).get("loss", float("nan")),
    }


def aggregate_run_summaries(spec: DatasetSpec, summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metric_name = primary_metric_name(spec)
    val_values = [
        float(summary.get("best_val", {}).get("metrics", {}).get(metric_name, float("nan")))
        for summary in summaries
    ]
    test_values = [
        float(summary.get("best_test", {}).get("metrics", {}).get(metric_name, float("nan")))
        for summary in summaries
    ]
    return {
        "num_runs": len(summaries),
        "primary_metric": metric_name,
        "higher_is_better": higher_is_better(spec),
        f"val_{metric_name}_mean": mean(val_values),
        f"val_{metric_name}_std": population_std(val_values),
        f"test_{metric_name}_mean": mean(test_values),
        f"test_{metric_name}_std": population_std(test_values),
    }


def aggregate_per_target_mae(
        summaries: Sequence[Dict[str, Any]],
        field_name: str = "per_target_mae") -> Dict[str, Any]:
    aggregated = {}
    for output_name, split_name in (("val", "best_val"), ("test", "best_test")):
        target_names = []
        for summary in summaries:
            for target_name in summary.get(split_name, {}).get(
                    field_name, {}):
                if target_name not in target_names:
                    target_names.append(target_name)
        if not target_names:
            continue
        aggregated[output_name] = {}
        for target_name in target_names:
            values = [
                float(summary.get(split_name, {}).get(
                    field_name, {}).get(target_name, float("nan")))
                for summary in summaries
            ]
            aggregated[output_name][target_name] = {
                "mean": mean(values),
                "std": population_std(values),
            }
    return aggregated


def format_metric_with_std(mean_value: float, std_value: float) -> str:
    return f"{float(mean_value):.4f} +/- {float(std_value):.4f}"


def format_paper_table_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    aggregate = summary["aggregate"]
    metric_name = aggregate["primary_metric"]
    direction = "higher" if aggregate.get("higher_is_better", True) else "lower"
    return {
        "dataset": summary["dataset"],
        "model": summary["model"],
        "metric": metric_name,
        "direction": direction,
        "runs": aggregate.get("num_runs", 0),
        "val": format_metric_with_std(
            aggregate.get(f"val_{metric_name}_mean", float("nan")),
            aggregate.get(f"val_{metric_name}_std", float("nan")),
        ),
        "test": format_metric_with_std(
            aggregate.get(f"test_{metric_name}_mean", float("nan")),
            aggregate.get(f"test_{metric_name}_std", float("nan")),
        ),
    }


def format_markdown_result_table(summaries: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Model | Metric | Direction | Runs | Val | Test |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for summary in summaries:
        row = format_paper_table_row(summary)
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['metric']} | {row['direction']} | "
            f"{row['runs']} | {row['val']} | {row['test']} |"
        )
    return "\n".join(lines)


def format_latex_result_rows(summaries: Sequence[Dict[str, Any]]) -> str:
    rows = []
    for summary in summaries:
        row = format_paper_table_row(summary)
        val = row["val"].replace("+/-", "$\\pm$")
        test = row["test"].replace("+/-", "$\\pm$")
        rows.append(
            f"{row['dataset']} & {row['model']} & {row['metric']} & {row['direction']} & "
            f"{row['runs']} & {val} & {test} \\"
        )
    return "\n".join(rows)


def mean_absolute_error(y_true, y_pred) -> float:
    pairs = finite_numeric_pairs(y_true, y_pred)
    if not pairs:
        return float("nan")
    return sum(abs(true - pred) for true, pred in pairs) / len(pairs)


def average_mae(y_true, y_pred) -> float:
    true_rows = as_2d_float_rows(y_true)
    pred_rows = as_2d_float_rows(y_pred)
    if not true_rows:
        return float("nan")
    num_tasks = max(len(row) for row in true_rows)
    scores = []
    for task_index in range(num_tasks):
        task_true = [row[task_index] for row in true_rows if task_index < len(row)]
        task_pred = [row[task_index] for row in pred_rows if task_index < len(row)]
        pairs = finite_numeric_pairs(task_true, task_pred)
        if not pairs:
            continue
        scores.append(sum(abs(true - pred) for true, pred in pairs) / len(pairs))
    return sum(scores) / len(scores) if scores else float("nan")


def flatten_numeric(values) -> List[float]:
    values = to_list(values)
    if not is_sequence(values):
        return [float(values)]
    result = []
    for value in values:
        if is_sequence(value):
            result.extend(flatten_numeric(value))
        else:
            result.append(float(value))
    return result


def finite_numeric_pairs(y_true, y_pred) -> List[Tuple[float, float]]:
    pairs = []
    for true, pred in zip(flatten_numeric(y_true), flatten_numeric(y_pred)):
        if math.isnan(true) or math.isnan(pred):
            continue
        pairs.append((true, pred))
    return pairs


def as_2d_float_rows(values) -> List[List[float]]:
    values = to_list(values)
    if not values:
        return []
    if not is_sequence(values[0]):
        return [[float(value)] for value in values]
    return [[float(item) for item in row] for row in values]


def binary_roc_auc(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    positives = [score for label, score in zip(y_true, y_score) if int(label) == 1]
    negatives = [score for label, score in zip(y_true, y_score) if int(label) == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif math.isclose(pos, neg):
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def average_precision(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    ranked = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
    total_positives = sum(1 for _, label in ranked if int(label) == 1)
    if total_positives == 0:
        return float("nan")
    hits = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if int(label) == 1:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / total_positives


def accuracy_score(y_true, y_score, threshold: float = 0.5) -> float:
    pairs = finite_numeric_pairs(y_true, y_score)
    if not pairs:
        return float("nan")
    correct = 0
    for true, score in pairs:
        prediction = 1 if float(score) >= threshold else 0
        if prediction == int(true):
            correct += 1
    return correct / len(pairs)


def _ensure_local_package(package: str, package_path: Path) -> None:
    module = sys.modules.setdefault(package, types.ModuleType(package))
    module.__path__ = [str(package_path)]


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_tokenizer_classes():
    root = ensure_repo_on_path()
    _ensure_local_package("gammagl", root / "gammagl")
    _ensure_local_package("gammagl.transforms", root / "gammagl" / "transforms")
    tokenizer_module = _load_module(
        "gammagl.transforms.graph_tokenizer",
        root / "gammagl" / "transforms" / "graph_tokenizer.py",
    )
    bpe_module = _load_module(
        "gammagl.transforms.graph_bpe",
        root / "gammagl" / "transforms" / "graph_bpe.py",
    )
    return tokenizer_module.GraphTokenizer, bpe_module.GraphBPE


def load_model_class(model_name: str):
    model_name = resolve_model_name(model_name)
    root = ensure_repo_on_path()
    _ensure_local_package("gammagl", root / "gammagl")
    _ensure_local_package("gammagl.models", root / "gammagl" / "models")
    graph_bert = _load_module("gammagl.models.graph_bert", root / "gammagl" / "models" / "graph_bert.py")
    if model_name == "bert":
        return graph_bert.GraphBERT
    graph_gte = _load_module("gammagl.models.graph_gte", root / "gammagl" / "models" / "graph_gte.py")
    return graph_gte.GraphGTE


def patch_tlx_torch_named_members() -> None:
    try:
        import tensorlayerx.nn.core.core_torch as core_torch
    except Exception:
        return
    if getattr(core_torch.Module._named_members, "_graph_tokenizer_patch", False):
        return
    original = core_torch.Module._named_members

    def patched(self, get_members_fn, prefix="", recurse=True, remove_duplicate=True):
        return original(self, get_members_fn, prefix=prefix, recurse=recurse)

    patched._graph_tokenizer_patch = True
    core_torch.Module._named_members = patched


def model_kwargs(args, spec: DatasetSpec):
    if args.smoke:
        return {
            "vocab_size": args.vocab_size,
            "output_dim": spec.output_dim,
            "hidden_size": args.hidden_size or 24,
            "num_hidden_layers": args.num_hidden_layers or 1,
            "num_attention_heads": args.num_attention_heads or 4,
            "intermediate_size": args.intermediate_size or 48,
            "max_position_embeddings": args.max_length,
        }
    kwargs = {
        "vocab_size": args.vocab_size,
        "output_dim": spec.output_dim,
        "max_position_embeddings": args.max_length,
    }
    optional_values = {
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "intermediate_size": args.intermediate_size,
    }
    kwargs.update({key: value for key, value in optional_values.items() if value is not None})
    return kwargs


def load_graphs(args, spec: DatasetSpec):
    if args.smoke:
        return make_synthetic_graphs(spec)
    splits = load_benchmark_splits(args.data_root, spec)
    return splits["train"] + splits["val"] + splits["test"]


def fit_tokenizer(args, train_graphs: Sequence[SyntheticGraph]):
    GraphTokenizer, GraphBPE = load_tokenizer_classes()
    from gammagl.transforms.graph_serializer import (
        EulerianSerializer,
        FrequencyGuidedEulerianSerializer,
    )

    serialization = getattr(args, "serialization", "feuler")
    serializer_cls = {
        "feuler": FrequencyGuidedEulerianSerializer,
        "eulerian": EulerianSerializer,
    }.get(serialization)
    if serializer_cls is None:
        raise ValueError("serialization must be 'feuler' or 'eulerian'.")
    tokenizer = GraphTokenizer(
        serializer=serializer_cls(),
        bpe=GraphBPE(
            num_merges=args.num_merges,
            min_frequency=args.min_frequency,
            backend=args.bpe_backend,
        )
    )
    if getattr(args, "protocol", None) == "paper":
        cache_root = Path(
            getattr(args, "paper_cache_root", None)
            or (Path(args.data_root) / ".graph_tokenizer_cache")
        )
        digest = hashlib.sha256()
        cache_config = {
            "cache_version": 1,
            "dataset": str(args.dataset),
            "serialization": serialization,
            "num_merges": int(args.num_merges),
            "min_frequency": int(args.min_frequency),
            "bpe_backend": str(args.bpe_backend),
            "num_train_graphs": len(train_graphs),
        }
        digest.update(json.dumps(
            cache_config, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"))
        for graph in train_graphs:
            digest.update(pickle.dumps((
                to_list(graph.edge_index),
                to_list(graph.x),
                to_list(graph.edge_attr),
                int(graph.num_nodes),
            ), protocol=4))
        cache_key = digest.hexdigest()
        cache_path = cache_root / str(args.dataset) / cache_key / "tokenizer.pkl"
        if cache_path.is_file():
            try:
                with cache_path.open("rb") as handle:
                    cached = pickle.load(handle)
                if (
                        cached.get("cache_version") == 1
                        and cached.get("cache_key") == cache_key):
                    tokenizer = cached["tokenizer"]
                    tokenizer._cache_status = "hit"
                    tokenizer._cache_key = cache_key
                    tokenizer._cache_path = str(cache_path)
                    return tokenizer
            except (OSError, EOFError, AttributeError, KeyError, pickle.UnpicklingError):
                pass
    tokenizer.fit(
        train_graphs,
        graph_ids=[
            getattr(graph, "graph_tokenizer_id", index)
            for index, graph in enumerate(train_graphs)
        ],
    )
    if getattr(args, "protocol", None) == "paper":
        tokenizer._cache_status = "miss"
        tokenizer._cache_key = cache_key
        tokenizer._cache_path = str(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(
            f".{cache_path.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as handle:
            pickle.dump({
                "cache_version": 1,
                "cache_key": cache_key,
                "tokenizer": tokenizer,
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, cache_path)
    return tokenizer


def encode_graph_batch(graphs: Sequence[SyntheticGraph], tokenizer, max_length: int) -> Dict[str, List[List[float]]]:
    encoded = [tokenizer.encode_graph(graph).input_ids for graph in graphs]
    input_ids, attention_mask = tokenizer.pad_token_sequences(
        encoded, max_length=max_length)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": [graph.y for graph in graphs],
    }


def encode_graph_splits(
    splits: Dict[str, List[SyntheticGraph]],
    tokenizer,
    max_length: int,
) -> Dict[str, Dict[str, List[List[float]]]]:
    return {
        split_name: encode_graph_batch(graphs, tokenizer, max_length=max_length)
        for split_name, graphs in splits.items()
    }


def iter_token_batches(encoded_split: Dict[str, List[List[float]]], batch_size: int):
    total = len(encoded_split["input_ids"])
    batch_size = max(1, int(batch_size))
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield {
            "input_ids": encoded_split["input_ids"][start:end],
            "attention_mask": encoded_split["attention_mask"][start:end],
            "labels": encoded_split["labels"][start:end],
        }


def build_mlm_pretrain_split(
    encoded_split: Dict[str, List[List[float]]],
    tokenizer,
    mask_prob: float,
    seed: int,
) -> Dict[str, List[List[int]]]:
    masked_ids, mlm_labels = tokenizer.mask_token_sequences(
        encoded_split["input_ids"],
        mask_prob=mask_prob,
        seed=seed,
    )
    return {
        "input_ids": masked_ids,
        "attention_mask": encoded_split["attention_mask"],
        "mlm_labels": mlm_labels,
    }


def count_mlm_labels(mlm_labels: Sequence[Sequence[int]]) -> int:
    return sum(1 for row in mlm_labels for value in row if int(value) != -100)


def iter_mlm_batches(encoded_split: Dict[str, List[List[int]]], batch_size: int):
    total = len(encoded_split["input_ids"])
    batch_size = max(1, int(batch_size))
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield {
            "input_ids": encoded_split["input_ids"][start:end],
            "attention_mask": encoded_split["attention_mask"][start:end],
            "mlm_labels": encoded_split["mlm_labels"][start:end],
        }


def tensor_to_float(value) -> float:
    value = to_list(value)
    if is_sequence(value):
        flat = flatten_numeric(value)
        return float(flat[0]) if flat else float("nan")
    return float(value)


def logits_to_predictions(spec: DatasetSpec, logits) -> List[List[float]]:
    rows = as_2d_float_rows(logits)
    if spec.metric in {"rocauc", "ap"}:
        return [[1.0 / (1.0 + math.exp(-value)) for value in row] for row in rows]
    return rows


def supervised_loss(logits, labels, spec: DatasetSpec):
    import tensorlayerx as tlx

    if spec.metric in {"mae", "average_mae"}:
        return tlx.losses.mean_squared_error(logits, labels)
    return tlx.losses.sigmoid_cross_entropy(target=labels, output=logits)


def masked_language_modeling_loss(mlm_logits, mlm_labels):
    try:
        import torch

        if isinstance(mlm_logits, torch.Tensor):
            vocab_size = int(mlm_logits.shape[-1])
            return torch.nn.functional.cross_entropy(
                mlm_logits.reshape(-1, vocab_size),
                mlm_labels.reshape(-1).long(),
                ignore_index=-100,
            )
    except ImportError:
        pass
    raise RuntimeError("MLM pretraining currently requires TensorLayerX with the PyTorch backend.")


class GraphTokenMLMLoss:
    def __new__(cls, model):
        from tensorlayerx.model import WithLoss

        class _Loss(WithLoss):
            def __init__(self, backbone):
                super().__init__(backbone=backbone, loss_fn=None)

            def forward(self, input_ids, attention_mask, mlm_labels):
                outputs = self.backbone_network(input_ids, attention_mask=attention_mask)
                return masked_language_modeling_loss(outputs["mlm_logits"], mlm_labels)

        return _Loss(model)


class GraphTokenSupervisedLoss:
    def __new__(cls, model, spec: DatasetSpec):
        import tensorlayerx as tlx
        from tensorlayerx.model import WithLoss

        class _Loss(WithLoss):
            def __init__(self, backbone, dataset_spec):
                super().__init__(backbone=backbone, loss_fn=None)
                self.dataset_spec = dataset_spec

            def forward(self, input_ids, attention_mask, labels):
                outputs = self.backbone_network(input_ids, attention_mask=attention_mask)
                return supervised_loss(outputs["logits"], labels, self.dataset_spec)

        return _Loss(model, spec)


def train_one_epoch(model, train_one_step, encoded_train, spec: DatasetSpec, args) -> float:
    import tensorlayerx as tlx

    model.set_train()
    losses = []
    for batch in iter_token_batches(encoded_train, args.batch_size):
        if not batch["input_ids"]:
            continue
        loss = train_one_step(
            tlx.convert_to_tensor(batch["input_ids"], dtype=tlx.int64),
            tlx.convert_to_tensor(batch["attention_mask"], dtype=tlx.int64),
            tlx.convert_to_tensor(batch["labels"], dtype=tlx.float32),
        )
        losses.append(tensor_to_float(loss))
    return sum(losses) / len(losses) if losses else float("nan")


def evaluate_split(model, encoded_split, spec: DatasetSpec, args) -> Dict[str, Any]:
    import tensorlayerx as tlx

    model.set_eval()
    losses = []
    y_true = []
    y_pred = []
    for batch in iter_token_batches(encoded_split, args.batch_size):
        if not batch["input_ids"]:
            continue
        labels = tlx.convert_to_tensor(batch["labels"], dtype=tlx.float32)
        outputs = model(
            tlx.convert_to_tensor(batch["input_ids"], dtype=tlx.int64),
            attention_mask=tlx.convert_to_tensor(batch["attention_mask"], dtype=tlx.int64),
        )
        loss = supervised_loss(outputs["logits"], labels, spec)
        losses.append(tensor_to_float(loss))
        y_true.extend(batch["labels"])
        y_pred.extend(logits_to_predictions(spec, to_list(outputs["logits"])))
    metrics = compute_task_metrics(spec, y_true, y_pred) if y_true else {}
    return {
        "loss": sum(losses) / len(losses) if losses else float("nan"),
        "metrics": metrics,
        "num_graphs": len(encoded_split["labels"]),
    }


def run_mlm_pretrain(model, encoded_train, tokenizer, args) -> Dict[str, Any]:
    if args.pretrain_epoch <= 0 or not encoded_train["input_ids"]:
        return {"epochs_ran": 0, "final_loss": float("nan"), "history": []}

    import tensorlayerx as tlx
    from tensorlayerx.model import TrainOneStep

    special_tokens = tokenizer.special_tokens
    special_token_ids = {
        special_tokens.pad_token_id,
        special_tokens.cls_token_id,
        special_tokens.sep_token_id,
        special_tokens.component_sep_token_id,
    }
    optimizer = tlx.optimizers.Adam(lr=args.pretrain_lr, weight_decay=args.weight_decay)
    train_one_step = TrainOneStep(GraphTokenMLMLoss(model), optimizer, model.trainable_weights)

    history = []
    for epoch in range(1, args.pretrain_epoch + 1):
        model.set_train()
        masked_train = build_mlm_pretrain_split(
            encoded_train,
            tokenizer=tokenizer,
            mask_prob=args.mask_prob,
            seed=effective_seed(args.seed) + epoch,
        )
        num_mlm_labels = count_mlm_labels(masked_train["mlm_labels"])
        losses = []
        if num_mlm_labels > 0:
            for batch in iter_mlm_batches(masked_train, args.batch_size):
                loss = train_one_step(
                    tlx.convert_to_tensor(batch["input_ids"], dtype=tlx.int64),
                    tlx.convert_to_tensor(batch["attention_mask"], dtype=tlx.int64),
                    tlx.convert_to_tensor(batch["mlm_labels"], dtype=tlx.int64),
                )
                losses.append(tensor_to_float(loss))
        history.append(
            {
                "epoch": epoch,
                "loss": sum(losses) / len(losses) if losses else float("nan"),
                "num_mlm_labels": num_mlm_labels,
            }
        )

    return {
        "epochs_ran": len(history),
        "final_loss": history[-1]["loss"] if history else float("nan"),
        "history": history,
    }


def checkpoint_path(args, spec: DatasetSpec, run_index: int) -> Path:
    return (
        Path(args.output_dir)
        / "checkpoints"
        / spec.canonical_name
        / args.model
        / f"run_{run_index}_best.npz"
    )


def save_model_checkpoint(model, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path), format="npz_dict")
    return str(path)


def run_train_val_test(args):
    spec = resolve_dataset_spec(args.dataset)
    random.seed(effective_seed(args.seed))
    splits = load_benchmark_splits(args.data_root, spec)
    train_graphs = splits["train"]
    if not train_graphs:
        raise ValueError("GraphTokenizer training split cannot be empty.")
    tokenizer = fit_tokenizer(args, train_graphs)
    encoded_splits = encode_graph_splits(splits, tokenizer, max_length=args.max_length)

    patch_tlx_torch_named_members()
    import tensorlayerx as tlx
    from tensorlayerx.model import TrainOneStep

    Model = load_model_class(args.model)
    model = Model(**model_kwargs(args, spec))
    pretrain_summary = run_mlm_pretrain(model, encoded_splits["train"], tokenizer, args)
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
    loss_func = GraphTokenSupervisedLoss(model, spec)
    train_one_step = TrainOneStep(loss_func, optimizer, model.trainable_weights)

    history = []
    best_value = None
    best_epoch = None
    best_checkpoint_path = None
    best_model_state = None
    stale_epochs = 0
    metric_name = primary_metric_name(spec)
    run_index = int(getattr(args, "run", 0))
    for epoch in range(1, args.n_epoch + 1):
        train_loss = train_one_epoch(model, train_one_step, encoded_splits["train"], spec, args)
        val_result = evaluate_split(model, encoded_splits["val"], spec, args)
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_result,
        }
        history.append(entry)
        val_metric = val_result["metrics"].get(metric_name, float("nan"))
        if metric_is_better(spec, float(val_metric), best_value):
            best_value = float(val_metric)
            best_epoch = epoch
            stale_epochs = 0
            best_model_state = copy.deepcopy(model.state_dict())
            if args.save_checkpoint:
                best_checkpoint_path = save_model_checkpoint(model, checkpoint_path(args, spec, run_index))
        else:
            stale_epochs += 1
        if args.patience > 0 and stale_epochs >= args.patience:
            break

    best = select_best_epoch(spec, history) or (history[-1] if history else None)
    if best_model_state is None:
        raise RuntimeError("No validation checkpoint was selected.")
    model.load_state_dict(best_model_state)
    best_test = evaluate_split(model, encoded_splits["test"], spec, args)
    return {
        "dataset": spec.canonical_name,
        "model": args.model,
        "run": run_index,
        "seed": effective_seed(args.seed),
        "metric": spec.metric,
        "primary_metric": metric_name,
        "higher_is_better": higher_is_better(spec),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "epochs_ran": len(history),
        "best_epoch": best_epoch if best_epoch is not None else (best["epoch"] if best else None),
        "pretrain": pretrain_summary,
        "checkpoint_path": best_checkpoint_path,
        "best_val": best["val"] if best else {},
        "best_test": best_test,
        "history": history,
    }


def clone_args_for_run(args, run_index: int):
    run_args = argparse.Namespace(**vars(args))
    run_args.run = run_index
    run_args.seed = effective_seed(args.seed) + int(run_index)
    return run_args


def parse_csv_values(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def experiment_matrix_items(args) -> List[Tuple[str, str]]:
    datasets = parse_csv_values(getattr(args, "datasets", None)) or [args.dataset]
    models = parse_csv_values(getattr(args, "models", None)) or [args.model]
    return [(dataset, model) for dataset in datasets for model in models]


def clone_args_for_experiment(args, dataset: str, model: str):
    experiment_args = argparse.Namespace(**vars(args))
    experiment_args.dataset = dataset
    experiment_args.model = model
    return experiment_args



def serialize_arg_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [serialize_arg_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_arg_value(item) for key, item in value.items()}
    return str(value)


def serialize_args(args) -> Dict[str, Any]:
    arg_dict = vars(args) if hasattr(args, "__dict__") else {}
    if arg_dict:
        items = arg_dict.items()
    else:
        items = (
            (name, getattr(args, name))
            for name in dir(args)
            if not name.startswith("_") and not callable(getattr(args, name))
        )
    return {name: serialize_arg_value(value) for name, value in sorted(items)}


def package_versions(names: Sequence[str]) -> Dict[str, Optional[str]]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_metadata() -> Dict[str, Any]:
    return {
        "status": "disabled",
        "reason": "local-only run; git commands are disabled",
    }


def build_runtime_manifest(args, timestamp: Optional[str] = None) -> Dict[str, Any]:
    return {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "args": serialize_args(args),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": package_versions(("numpy", "torch", "torchvision", "tensorlayerx", "gammagl")),
        "git": git_metadata(),
    }



def parse_args(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.model is None:
        args.model = "gte" if args.protocol == "paper" else "bert"
    return args



def preflight_dataset_status(data_root, dataset: str) -> Dict[str, Any]:
    spec = resolve_dataset_spec(dataset)
    status: Dict[str, Any] = {
        "dataset": spec.canonical_name,
        "paper_name": spec.paper_name,
        "status": "ok",
    }
    gammagl_dataset = load_gammagl_benchmark_dataset(data_root, spec)
    if gammagl_dataset is None:
        raise FileNotFoundError(
            f"GammaGL dataset loader could not materialize {spec.paper_name}.")
    split_indices = gammagl_dataset.get_idx_split()
    validate_split_indices(len(gammagl_dataset), split_indices)
    status.update(
        {
            "dataset_dir": str(gammagl_dataset.root_dir),
            "storage_dir": str(gammagl_dataset.raw_dir),
            "num_samples": len(gammagl_dataset),
            "split_sizes": {split_name: len(indices) for split_name, indices in split_indices.items()},
            "downloaded_by_dataset": True,
        }
    )
    return status


def preflight_model_status(model: str) -> Dict[str, Any]:
    model_name = resolve_model_name(model)
    filename = "graph_bert.py" if model_name == "bert" else "graph_gte.py"
    model_path = repo_root() / "gammagl" / "models" / filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    status = {"model": model_name, "status": "ok", "file": str(model_path)}
    if model_name == "gte":
        status.update({
            "reproduction_ready": False,
            "reason": (
                "GraphGTE is randomly initialized; official "
                "Alibaba-NLP/gte-multilingual-base loading is not implemented."),
        })
    return status


def preflight_bpe_backend_status(backend: str) -> Dict[str, Any]:
    ensure_repo_on_path()
    if backend not in {"python", "auto", "cpp"}:
        raise ValueError("bpe_backend must be one of: python, auto, cpp.")
    status = {"backend": backend, "status": "ok"}
    if backend == "cpp":
        from third_party import graph_bpe_cpp

        if not graph_bpe_cpp.is_available():
            raise ImportError("GraphBPE backend='cpp' requires the optional graph_bpe_cpp native extension.")
        status["native_available"] = True
    elif backend == "auto":
        from third_party import graph_bpe_cpp

        status["native_available"] = graph_bpe_cpp.is_available()
    return status


def preflight_check(args) -> Dict[str, Any]:
    datasets = parse_csv_values(getattr(args, "datasets", None)) or [args.dataset]
    models = parse_csv_values(getattr(args, "models", None))
    if not models:
        models = [args.model or "gte"]
    report: Dict[str, Any] = {
        "status": "ok",
        "data_root": str(getattr(args, "data_root", "data")),
        "datasets": [],
        "models": [],
        "bpe_backend": None,
        "paper_runtime": None,
        "errors": [],
    }
    if getattr(args, "protocol", "gammagl") == "paper":
        try:
            report["paper_runtime"] = (
                load_paper_protocol_module().paper_runtime_status())
        except Exception as error:
            report["paper_runtime"] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            report["errors"].append(str(error))

    for dataset in datasets:
        try:
            if getattr(args, "protocol", "gammagl") == "paper":
                protocol = load_paper_protocol_module()
                protocol.validate_paper_args(args, resolve_dataset_spec(dataset))
            report["datasets"].append(preflight_dataset_status(args.data_root, dataset))
        except Exception as error:
            report["datasets"].append(
                {
                    "dataset": str(dataset),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            report["errors"].append(str(error))

    for model in models:
        try:
            report["models"].append(preflight_model_status(model))
        except Exception as error:
            report["models"].append(
                {
                    "model": str(model),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            report["errors"].append(str(error))

    try:
        report["bpe_backend"] = preflight_bpe_backend_status(args.bpe_backend)
    except Exception as error:
        report["bpe_backend"] = {
            "backend": str(getattr(args, "bpe_backend", None)),
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        report["errors"].append(str(error))

    report["num_errors"] = len(report["errors"])
    if report["num_errors"]:
        report["status"] = "failed"
    return report

def experiment_summary_path(args, dataset: str, model: str) -> Path:
    spec = resolve_dataset_spec(dataset)
    return Path(args.output_dir) / f"{spec.canonical_name}_{model}_summary.json"


def load_existing_experiment_summary(args, dataset: str, model: str) -> Optional[Dict[str, Any]]:
    path = experiment_summary_path(args, dataset, model)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    summary.setdefault("status", "skipped_existing")
    summary.setdefault("outputs", {})["json"] = str(path)
    return summary


def format_failed_experiment_summary(dataset: str, model: str, error: Exception) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "model": model,
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "runs": [],
        "rows": [],
        "aggregate": {
            "num_runs": 0,
            "primary_metric": None,
            "higher_is_better": None,
        },
    }


def write_experiment_outputs(summary: Dict[str, Any], args) -> Dict[str, str]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = summary["dataset"]
    model = summary["model"]
    json_path = output_dir / f"{dataset}_{model}_summary.json"
    csv_path = output_dir / f"{dataset}_{model}_runs.csv"
    markdown_path = output_dir / f"{dataset}_{model}_paper_table.md"
    latex_path = output_dir / f"{dataset}_{model}_paper_table.tex"
    manifest_path = output_dir / f"{dataset}_{model}_manifest.json"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    rows = summary.get("rows", [])
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    table_summaries = [summary]
    markdown_path.write_text(format_markdown_result_table(table_summaries), encoding="utf-8")
    latex_path.write_text(format_latex_result_rows(table_summaries), encoding="utf-8")
    if "manifest" in summary:
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(summary["manifest"], handle, indent=2)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "latex": str(latex_path),
        "manifest": str(manifest_path),
    }


def run_experiments(args):
    spec = resolve_dataset_spec(args.dataset)
    runs = [
        run_train_val_test(clone_args_for_run(args, run_index))
        for run_index in range(max(1, int(args.runs or 1)))
    ]
    rows = [format_result_row(summary) for summary in runs]
    summary = {
        "dataset": spec.canonical_name,
        "model": args.model,
        "status": "completed",
        "runs": runs,
        "rows": rows,
        "aggregate": aggregate_run_summaries(spec, runs),
        "manifest": build_runtime_manifest(args),
    }
    summary["outputs"] = write_experiment_outputs(summary, args)
    return summary


def load_paper_protocol_module():
    module_name = "graph_tokenizer_paper_protocol"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = Path(__file__).with_name("paper_protocol.py")
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


def run_paper_experiments(args):
    spec = resolve_dataset_spec(args.dataset)
    protocol = load_paper_protocol_module()
    protocol.validate_paper_args(args, spec)
    paper_summary = protocol.run_paper_experiment(
        args,
        spec,
        load_benchmark_splits(args.data_root, spec),
        fit_tokenizer,
    )
    primary_metric = "rocauc" if spec.canonical_name == "molhiv" else spec.metric
    runs = []
    for run in paper_summary["runs"]:
        runs.append({
            "dataset": spec.canonical_name,
            "model": paper_summary["model"],
            "run": run["run"],
            "seed": run["seed"],
            "best_epoch": run["best_epoch"],
            "best_val": {
                "loss": run["best_val"]["loss"],
                "metrics": {primary_metric: run["best_val"]["metric"]},
                "metric_space": run["best_val"].get("metric_space"),
                "per_target_mae": run["best_val"].get(
                    "per_target_mae", {}),
                "per_target_mae_raw": run["best_val"].get(
                    "per_target_mae_raw", {}),
            },
            "best_test": {
                "loss": run["best_test"]["loss"],
                "metrics": {primary_metric: run["best_test"]["metric"]},
                "metric_space": run["best_test"].get("metric_space"),
                "per_target_mae": run["best_test"].get(
                    "per_target_mae", {}),
                "per_target_mae_raw": run["best_test"].get(
                    "per_target_mae_raw", {}),
            },
            "checkpoint_path": run["checkpoint_path"],
            "history": run["history"],
            "test_evaluations": run["test_evaluations"],
            "model_manifest": run["model"],
        })
    values = [run["best_test"]["metrics"][primary_metric] for run in runs]
    val_values = [run["best_val"]["metrics"][primary_metric] for run in runs]
    summary = {
        "dataset": spec.canonical_name,
        "model": paper_summary["model"],
        "protocol": "paper",
        "status": "completed",
        "runs": runs,
        "rows": [format_result_row({**run, "primary_metric": primary_metric}) for run in runs],
        "aggregate": {
            "num_runs": len(runs),
            "primary_metric": primary_metric,
            "higher_is_better": spec.canonical_name == "molhiv",
            f"val_{primary_metric}_mean": mean(val_values),
            f"val_{primary_metric}_std": population_std(val_values),
            f"test_{primary_metric}_mean": mean(values),
            f"test_{primary_metric}_std": population_std(values),
            "per_target_mae": aggregate_per_target_mae(runs),
            "per_target_mae_raw": aggregate_per_target_mae(
                runs, field_name="per_target_mae_raw"),
        },
        "paper": {key: value for key, value in paper_summary.items() if key != "runs"},
        "manifest": build_runtime_manifest(args),
    }
    summary["outputs"] = write_experiment_outputs(summary, args)
    return summary


def write_experiment_matrix_outputs(summary: Dict[str, Any], args) -> Dict[str, str]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "graph_tokenizer_all_summary.json"
    csv_path = output_dir / "graph_tokenizer_all_runs.csv"
    markdown_path = output_dir / "graph_tokenizer_all_paper_table.md"
    latex_path = output_dir / "graph_tokenizer_all_paper_table.tex"
    manifest_path = output_dir / "graph_tokenizer_all_manifest.json"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    all_rows = []
    for experiment in summary.get("experiments", []):
        all_rows.extend(experiment.get("rows", []))
    if all_rows:
        fieldnames = sorted({key for row in all_rows for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    experiments = summary.get("experiments", [])
    markdown_path.write_text(format_markdown_result_table(experiments), encoding="utf-8")
    latex_path.write_text(format_latex_result_rows(experiments), encoding="utf-8")
    if "manifest" in summary:
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(summary["manifest"], handle, indent=2)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "latex": str(latex_path),
        "manifest": str(manifest_path),
    }


def run_experiment_matrix(args):
    experiments = []
    failures = []
    for dataset, model in experiment_matrix_items(args):
        if getattr(args, "resume", False):
            existing = load_existing_experiment_summary(args, dataset, model)
            if existing is not None:
                experiments.append(existing)
                continue
        try:
            experiments.append(run_experiments(clone_args_for_experiment(args, dataset, model)))
        except Exception as error:
            failure = format_failed_experiment_summary(dataset, model, error)
            failures.append(failure)
            experiments.append(failure)
            if not getattr(args, "keep_going", False):
                summary = {
                    "datasets": parse_csv_values(getattr(args, "datasets", None)) or [args.dataset],
                    "models": parse_csv_values(getattr(args, "models", None)) or [args.model],
                    "num_experiments": len(experiments),
                    "num_failures": len(failures),
                    "experiments": experiments,
                    "failures": failures,
                    "manifest": build_runtime_manifest(args),
                }
                summary["outputs"] = write_experiment_matrix_outputs(summary, args)
                raise
    summary = {
        "datasets": parse_csv_values(getattr(args, "datasets", None)) or [args.dataset],
        "models": parse_csv_values(getattr(args, "models", None)) or [args.model],
        "num_experiments": len(experiments),
        "num_failures": len(failures),
        "experiments": experiments,
        "failures": failures,
        "manifest": build_runtime_manifest(args),
    }
    summary["outputs"] = write_experiment_matrix_outputs(summary, args)
    return summary


def run_smoke(args):
    spec = resolve_dataset_spec(args.dataset)
    graphs = load_graphs(args, spec)
    tokenizer = fit_tokenizer(args, graphs)
    encoded = [tokenizer.encode_graph(graph).input_ids for graph in graphs]
    mlm_batch = tokenizer.build_mlm_batch_from_token_sequences(
        encoded,
        max_length=args.max_length,
        mask_prob=args.mask_prob,
        seed=effective_seed(args.seed),
    )
    masked_ids = mlm_batch.input_ids
    attention_mask = mlm_batch.attention_mask
    mlm_labels = mlm_batch.labels

    patch_tlx_torch_named_members()
    import tensorlayerx as tlx

    Model = load_model_class(args.model)
    model = Model(**model_kwargs(args, spec))
    outputs = model(
        tlx.convert_to_tensor(masked_ids, dtype=tlx.int64),
        attention_mask=tlx.convert_to_tensor(attention_mask, dtype=tlx.int64),
    )
    labels = [graph.y for graph in graphs]
    zero_predictions = [[0.0] * spec.output_dim for _ in graphs]
    metrics = compute_task_metrics(spec, labels, zero_predictions)
    return {
        "dataset": spec.canonical_name,
        "model": args.model,
        "num_graphs": len(graphs),
        "input_shape": tlx.get_tensor_shape(outputs["last_hidden_state"])[:2],
        "logits_shape": tlx.get_tensor_shape(outputs["logits"]),
        "mlm_logits_shape": tlx.get_tensor_shape(outputs["mlm_logits"]),
        "num_mlm_labels": sum(1 for row in mlm_labels for value in row if value != -100),
        "metrics": metrics,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphTokenizer training and smoke-test entrypoint.")
    parser.add_argument(
        "--dataset",
        default="qm9",
        help="One of: QM9, OGBG-molhiv/molhiv, Peptides-func/p-func, Peptides-struct/p-struct.",
    )
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset matrix, e.g. qm9,molhiv,p-func,p-struct.")
    parser.add_argument("--model", default=None, choices=SUPPORTED_MODELS)
    parser.add_argument("--models", default=None, help="Comma-separated model matrix, e.g. bert,gte.")
    parser.add_argument("--protocol", default="gammagl", choices=("gammagl", "paper"))
    parser.add_argument(
        "--paper-cache-root",
        default=None,
        help="Tokenizer/serialization cache root.",
    )
    parser.add_argument(
        "--paper-amp", default=None, choices=("off", "fp16", "bf16"),
        help="Optional CUDA autocast mode; strict paper default is off.",
    )
    parser.add_argument(
        "--paper-tf32", action=argparse.BooleanOptionalAction, default=None,
        help="Enable TF32 matmul explicitly; strict paper default is disabled.",
    )
    parser.add_argument(
        "--paper-loss", default=None,
        choices=("default", "l1", "huber", "focal"),
        help="Optional supervised loss ablation; default preserves the paper loss.",
    )
    parser.add_argument(
        "--molhiv-pos-weight", action=argparse.BooleanOptionalAction,
        default=None,
        help="Weight molhiv positives from training labels only.",
    )
    parser.add_argument(
        "--target-property",
        default=None,
        help="Legacy option rejected by strict paper mode, which jointly predicts all 16 QM9 targets.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device used by --protocol paper.")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="logs/graph_tokenizer")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--pretrain-epoch", type=int, default=1)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--pretrain-warmup-ratio", type=float, default=0.12)
    parser.add_argument("--pretrain-max-grad-norm", type=float, default=2.0)
    parser.add_argument("--n-epoch", type=int, default=1)
    parser.add_argument("--finetune-warmup-ratio", type=float, default=0.025)
    parser.add_argument("--finetune-max-grad-norm", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--pooling", default="mean", choices=("mean", "cls"))
    parser.add_argument("--num-merges", type=int, default=2000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--bpe-backend", default="auto", choices=("python", "auto", "cpp"))
    parser.add_argument("--serialization", default="feuler", choices=("feuler", "eulerian"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-hidden-layers", type=int, default=None)
    parser.add_argument("--num-attention-heads", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--max-position-embeddings", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-save-checkpoint", dest="save_checkpoint", action="store_false")
    parser.add_argument("--resume", action="store_true", help="Skip dataset/model matrix entries whose summary JSON already exists.")
    parser.add_argument("--keep-going", action="store_true", help="Record failed matrix entries and continue with the remaining entries.")
    parser.add_argument("--preflight", action="store_true", help="Validate datasets, model names, and BPE backend without training.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny synthetic end-to-end pipeline.")
    parser.add_argument("--list-datasets", action="store_true")
    parser.set_defaults(save_checkpoint=True)
    return parser


def main(argv=None):
    args = parse_args(argv)
    if args.list_datasets:
        for spec in DATASET_SPECS:
            print(f"{spec.paper_name}: aliases={','.join(spec.aliases)} metric={spec.metric}")
        return None
    if args.preflight:
        report = preflight_check(args)
        print(json.dumps(report, indent=2))
        if report["status"] != "ok":
            raise SystemExit(1)
        return report
    if args.smoke:
        if args.protocol == "paper":
            raise ValueError("Paper protocol does not support synthetic --smoke data.")
        summary = run_smoke(args)
        print(summary)
        return summary
    if args.protocol == "paper":
        if args.datasets or args.models:
            raise ValueError(
                "Paper protocol runs one dataset and encoder at a time; "
                "use --dataset and optional --model.")
        summary = run_paper_experiments(args)
        print(summary)
        return summary
    if args.datasets or args.models:
        summary = run_experiment_matrix(args)
    else:
        summary = run_experiments(args)
    print(summary)
    return summary


if __name__ == "__main__":
    main()
