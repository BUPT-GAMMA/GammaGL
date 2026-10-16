import importlib.util
import io
import json
import math
import os
import pickle
import random
import shutil
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def trainer():
    spec = importlib.util.spec_from_file_location(
        "graph_tokenizer_pipeline_trainer", ROOT / "examples" / "graph_tokenizer" / "graph_tokenizer_trainer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _aqsol_graphs(trainer, count, offset=0):
    return [trainer.GraphRecord(
        [[0, 1], [1, 2]], [index % 4 + 1, 2, 3], [1, 2],
        [float(index)])
        for index in range(offset, offset + count)]


def _preprocessing_args(root, cache):
    return SimpleNamespace(
        dataset="aqsol", encoder="bert", data_root=str(root),
        preprocessing_cache_dir=str(cache), bpe_merges=2,
        bpe_min_frequency=2, bpe_backend="python", num_serializations=1,
        max_length=64)


def _write_preprocessing_raw(root, first_feature=1, first_label=0.0):
    raw = root / "aqsol" / "raw"
    raw.mkdir(parents=True)
    samples = [
        ({"edge_index": [[0], [1]], "x": [first_feature if index == 0 else index + 1, 2],
          "edge_attr": [1]}, first_label if index == 0 else float(index))
        for index in range(5)
    ]
    with (raw / "data.pkl").open("wb") as handle:
        pickle.dump(samples, handle)
    for name, indices in (("train", [0, 1, 2]), ("val", [3]), ("test", [4])):
        (raw / f"{name}_index.json").write_text(json.dumps(indices), encoding="utf-8")


def _cache_events(capsys):
    return [json.loads(line)["status"] for line in capsys.readouterr().out.splitlines()
            if '"event": "preprocessing_cache"' in line]


def _record(graph_id, labels):
    return {"input_ids": [3, 8, 4], "labels": labels, "graph_id": graph_id}


class FixedLengthTokenizer:
    def __init__(self, input_ids):
        from gammagl.transforms.graph_tokenizer import GraphTokenizer

        self.input_ids = input_ids
        self.special_tokens = SimpleNamespace(sep_token_id=4)
        self.strict_tokenizer = GraphTokenizer()

    @staticmethod
    def realization_start_nodes(_graph, _num_serializations):
        return [None]

    def encode_graph(self, _graph, start_node=None):
        return SimpleNamespace(input_ids=list(self.input_ids),
                               serialized_token_ids=list(self.input_ids[1:-1]))

    def validate_token_sequences(self, token_sequences, max_length):
        return self.strict_tokenizer.validate_token_sequences(token_sequences, max_length)


class FakeDGLGraph:
    """Pickleable subset of the DGL graph API used by the release adapters."""

    def __init__(self, src, dst, ndata, edata):
        self._src = src
        self._dst = dst
        self.ndata = ndata
        self.edata = edata

    def edges(self):
        return self._src, self._dst


def _write_graph_tokenizer_bundle(path):
    datasets = {
        # The official Mutagenicity and DBLP release loaders accept DGL graphs
        # with node_token_ids and synthesize their documented zero edge token.
        "mutagenicity": [(FakeDGLGraph([0], [1], {"node_token_ids": [[5], [9]]}, {}), 1)],
        "dblp": [(FakeDGLGraph([0], [1], {"node_token_ids": [[7], [11]]}, {}), 0)],
        # AQSOL uses DGL ndata/edata feat fields; both are mandatory and mapped
        # by its adapter using the official atomic-number/bond-type token rules.
        "aqsol": [(FakeDGLGraph([0], [1], {"feat": [6, 8]}, {"feat": [1]}), -2.75)],
    }
    with tarfile.open(path, "w:gz") as archive:
        for dataset_name, samples in datasets.items():
            files = {
                f"release/{dataset_name}/data.pkl": pickle.dumps(samples),
                f"release/{dataset_name}/train_index.json": b"[0]",
                f"release/{dataset_name}/val_index.json": b"[]",
                f"release/{dataset_name}/test_index.json": b"[]",
            }
            for name, content in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))


class FixedLogits(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.as_tensor(logits, dtype=torch.float32))

    def forward(self, input_ids, attention_mask=None, task=None):
        return self.logits[:len(input_ids)]


def test_prepare_data_mode_materializes_shared_bundle_and_training_reuses_it(
        trainer, tmp_path, monkeypatch, capsys):
    from gammagl.datasets import _graph_tokenizer_download as download

    bundle = tmp_path / "bundle.tar.gz"
    _write_graph_tokenizer_bundle(bundle)
    monkeypatch.setattr(download, "PAPER_DATA_BUNDLE_SHA256", download.sha256_file(bundle))
    monkeypatch.setenv(download.DATA_BUNDLE_ENV, str(bundle))
    monkeypatch.setattr("gammagl.data.download.download_google_url",
                        lambda *_args, **_kwargs: pytest.fail("prepare mode must reuse its local bundle"))
    root = tmp_path / "data"

    assert trainer.main(["--prepare-data", "--data-root", str(root)]) is None
    first_bytes = (root / "aqsol" / "raw" / "data.pkl").read_bytes()
    output = capsys.readouterr().out
    assert "mutagenicity: READY" in output
    assert "dblp: READY" in output
    assert "aqsol: READY" in output

    assert trainer.main(["--prepare-data", "--data-root", str(root)]) is None
    assert (root / "aqsol" / "raw" / "data.pkl").read_bytes() == first_bytes
    assert trainer.load_dataset_splits(str(root), trainer.resolve_dataset("aqsol"))["train"]


def test_training_requires_prepared_data_without_network(trainer, tmp_path, monkeypatch):
    monkeypatch.delenv("GAMMAGL_GRAPH_TOKENIZER_DATA_BUNDLE", raising=False)
    monkeypatch.setattr("gammagl.data.download.download_google_url",
                        lambda *_args, **_kwargs: pytest.fail("training must not download the shared bundle"))

    with pytest.raises(FileNotFoundError, match="GraphTokenizer dataset is not prepared"):
        trainer.load_dataset_splits(str(tmp_path / "empty"), trainer.resolve_dataset("aqsol"))


@pytest.mark.parametrize("direction", [{"directed": True}, {"is_directed": True}])
def test_trainer_adapter_rejects_explicit_directed_graphs(trainer, direction):
    from gammagl.data import Graph

    graph = Graph(edge_index=[[0], [1]], x=[5, 9], edge_attr=[1],
                  y=[1.0], **direction)
    with pytest.raises(ValueError, match="directed graph semantics"):
        trainer.graph_from_gammagl(graph, trainer.resolve_dataset("aqsol"))


@pytest.mark.parametrize("edges,direction", [
    ([[0], [1]], {"directed": False}),
    ([[0], [1]], {}),
    ([[0, 1], [1, 0]], {}),
])
def test_trainer_adapter_accepts_undirected_coo_storage(trainer, edges, direction):
    from gammagl.data import Graph
    from gammagl.transforms.graph_serializer import FrequencyGuidedEulerianSerializer

    graph = Graph(edge_index=edges, x=[5, 9], edge_attr=[1] * len(edges[0]),
                  y=[1.0], **direction)
    record = trainer.graph_from_gammagl(graph, trainer.resolve_dataset("aqsol"))
    assert FrequencyGuidedEulerianSerializer().serialize(record).token_ids


@pytest.mark.parametrize("split_name", ("train", "val", "test"))
def test_encode_splits_rejects_overlong_sequences_for_every_split(trainer, split_name):
    args = SimpleNamespace(dataset="aqsol", num_serializations=1, max_length=8)
    tokenizer = FixedLengthTokenizer([3, 8, 9, 10, 11, 12, 13, 14, 4])
    graph = SimpleNamespace(y=[0.0])

    with pytest.raises(ValueError) as error:
        trainer.encode_splits(tokenizer, {split_name: [graph]}, args)

    message = str(error.value)
    assert "dataset=aqsol" in message
    assert f"split={split_name}" in message
    assert "sample=0" in message
    assert "length=9" in message
    assert "max_length=8" in message


def test_encode_splits_allows_sequence_at_max_length(trainer):
    args = SimpleNamespace(dataset="aqsol", num_serializations=1, max_length=8)
    tokenizer = FixedLengthTokenizer([3, 8, 9, 10, 11, 12, 13, 4])
    graph = SimpleNamespace(y=[0.0])

    encoded = trainer.encode_splits(tokenizer, {"train": [graph]}, args)

    assert encoded["train"][0]["input_ids"] == [3, 8, 9, 10, 11, 12, 13, 4]


@pytest.mark.parametrize("dataset,length", (
    ("mutagenicity", 768),
    ("dblp", 826),
    ("aqsol", 1587),
    ("aqsol", 8096),
))
def test_bert_formal_presets_accept_sequences_up_to_8096(trainer, dataset, length):
    args = trainer.apply_preset(trainer.build_parser().parse_args([
        "--dataset", dataset, "--encoder", "bert",
    ]))
    tokenizer = FixedLengthTokenizer([3] + list(range(8, 8 + length - 2)) + [4])
    label_width = trainer.resolve_dataset(dataset).label_width
    graph = SimpleNamespace(y=[0.0] * label_width)

    encoded = trainer.encode_splits(tokenizer, {"train": [graph]}, args)

    assert args.max_length == 8096
    assert len(encoded["train"][0]["input_ids"]) == length


def test_bert_formal_preset_rejects_sequence_longer_than_8096(trainer):
    args = trainer.apply_preset(trainer.build_parser().parse_args([
        "--dataset", "mutagenicity", "--encoder", "bert",
    ]))
    tokenizer = FixedLengthTokenizer([3] + list(range(8, 8 + 8097 - 2)) + [4])
    graph = SimpleNamespace(y=[0.0])

    with pytest.raises(ValueError, match="max_length=8096"):
        trainer.encode_splits(tokenizer, {"train": [graph]}, args)


def test_preset_rejects_a_sequence_limit_above_positional_capacity(trainer):
    args = trainer.build_parser().parse_args([
        "--dataset", "aqsol", "--encoder", "bert", "--max-length", "8097",
    ])

    with pytest.raises(ValueError, match="max_length must not exceed max_position_embeddings"):
        trainer.apply_preset(args)


def test_trainer_collate_pads_to_the_longest_sequence_in_its_batch(trainer):
    records = [
        {"input_ids": list(range(826)), "labels": [0.0], "graph_id": 0},
        {"input_ids": list(range(7)), "labels": [1.0], "graph_id": 1},
    ]
    loader = trainer.make_loader(
        torch, records, batch_size=2, pad_token_id=0, shuffle=False,
        choose_variant=False)

    input_ids, attention, _, _ = next(iter(loader))

    assert input_ids.shape == (2, 826)
    assert attention.shape == (2, 826)


def _run_result(seed, run_index, metric):
    return {
        "dataset": "aqsol",
        "encoder": "bert",
        "seed": seed,
        "run_index": run_index,
        "metric_name": "MAE",
        "metric": metric,
    }


def test_default_seeds_are_five_explicit_runs(trainer):
    assert trainer.build_parser().parse_args([]).seeds == [42, 43, 44, 45, 46]


def test_paper_preset_uses_phase_specific_warmup_and_gradient_clipping(trainer):
    args = trainer.apply_preset(trainer.build_parser().parse_args([
        "--dataset", "mutagenicity", "--encoder", "gte",
    ]))

    assert args.pretrain_learning_rate == 1e-4
    assert args.learning_rate == 5e-5
    assert args.pretrain_epochs == 200
    assert args.mask_probability == 0.09
    assert args.pretrain_warmup_ratio == 0.12
    assert args.finetune_warmup_ratio == 0.025
    assert args.pretrain_max_grad_norm == 2.0
    assert args.finetune_max_grad_norm == 0.5
    assert args.pretrain_batch_size == 32
    assert args.finetune_batch_size == 64
    assert (args.pretrain_swap_probability, args.pretrain_swap_ratio,
            args.pretrain_swap_window) == (0.5, 0.10, 3)
    assert (args.finetune_swap_probability, args.finetune_swap_ratio,
            args.finetune_swap_window) == (0.4, 0.05, 3)
    assert (args.finetune_mask_probability, args.finetune_mask_ratio) == (0.3, 0.05)
    assert (args.finetune_noise_probability, args.finetune_noise_std) == (0.3, 0.01)


@pytest.mark.parametrize(("argv", "expected"), [
    ([], (32, 32)),
    (["--batch-size", "7"], (7, 7)),
    (["--batch-size", "8", "--finetune-batch-size", "4"], (8, 4)),
    (["--batch-size", "8", "--pretrain-batch-size", "2"], (2, 8)),
])
def test_batch_size_overrides_presets_without_overriding_phase_specific_values(
        trainer, argv, expected):
    args = trainer.apply_preset(trainer.build_parser().parse_args([
        "--dataset", "mutagenicity", "--encoder", "bert", *argv,
    ]))

    assert (args.pretrain_batch_size, args.finetune_batch_size) == expected


def test_dblp_gte_preset_uses_its_official_finetune_batch_size(trainer):
    args = trainer.apply_preset(trainer.build_parser().parse_args([
        "--dataset", "dblp", "--encoder", "gte",
    ]))

    assert args.pretrain_batch_size == 32
    assert args.finetune_batch_size == 64
    assert args.gradient_accumulation_steps == 1


def test_preprocessing_cache_reuses_fitted_tokenizer_and_encoded_splits(
        trainer, tmp_path, monkeypatch):
    args = SimpleNamespace(
        dataset="aqsol", encoder="bert", data_root=str(tmp_path / "data"),
        preprocessing_cache_dir=str(tmp_path / "cache"), bpe_merges=2,
        bpe_min_frequency=2, bpe_backend="python", num_serializations=1,
        max_length=64)
    spec = trainer.resolve_dataset("aqsol")
    splits = {"train": _aqsol_graphs(trainer, 3),
              "val": _aqsol_graphs(trainer, 1, offset=3),
              "test": _aqsol_graphs(trainer, 1, offset=4)}
    monkeypatch.setattr(trainer, "load_dataset_splits", lambda *_: splits)

    first_tokenizer, first_encoded, first_normalizer, first_output_dim = (
        trainer.load_or_prepare_preprocessing(args, spec))
    monkeypatch.setattr(trainer, "load_dataset_splits",
                        lambda *_: pytest.fail("cache hit must not reload raw data"))
    second_tokenizer, second_encoded, second_normalizer, second_output_dim = (
        trainer.load_or_prepare_preprocessing(args, spec))

    assert second_tokenizer.vocabulary == first_tokenizer.vocabulary
    assert second_encoded == first_encoded
    assert second_normalizer == first_normalizer
    assert second_output_dim == first_output_dim == 1
    assert len(list((tmp_path / "cache").glob("*.pickle"))) == 1


def test_first_processing_then_same_processed_content_hits_cache(
        trainer, tmp_path, monkeypatch, capsys):
    root = tmp_path / "data"
    _write_preprocessing_raw(root)
    args = _preprocessing_args(root, tmp_path / "shared-cache")
    spec = trainer.resolve_dataset("aqsol")

    first = trainer.load_or_prepare_preprocessing(args, spec)
    assert _cache_events(capsys) == ["created"]
    assert (root / "aqsol" / "processed" / "split_indices_v2.json").exists()
    assert first[2]["mean"] == [1.0]
    monkeypatch.setattr(trainer, "load_dataset_splits",
                        lambda *_: pytest.fail("cache hit must skip dataset loading"))

    second = trainer.load_or_prepare_preprocessing(args, spec)
    assert _cache_events(capsys) == ["hit"]
    assert second[1] == first[1]
    assert second[2] == first[2]
    (root / "aqsol" / "raw" / "train_index.json").write_text("[1, 2, 3]", encoding="utf-8")
    args.seed = 99
    assert trainer.load_or_prepare_preprocessing(args, spec)[1] == first[1]
    assert _cache_events(capsys) == ["hit"]
    assert len(list((tmp_path / "shared-cache").glob("*.pickle"))) == 1


@pytest.mark.parametrize("changed", ("graph", "label"))
def test_processed_only_content_change_rebuilds_preprocessing(
        trainer, tmp_path, capsys, changed):
    from gammagl.datasets import AQSOL
    import tensorlayerx as tlx

    root = tmp_path / "data"
    _write_preprocessing_raw(root)
    args = _preprocessing_args(root, tmp_path / "shared-cache")
    spec = trainer.resolve_dataset("aqsol")
    first = trainer.load_or_prepare_preprocessing(args, spec)
    assert _cache_events(capsys) == ["created"]
    shutil.rmtree(root / "aqsol" / "raw")
    assert trainer.load_or_prepare_preprocessing(args, spec)[1] == first[1]
    assert _cache_events(capsys) == ["hit"]
    dataset = AQSOL(root=str(root))
    graphs = [dataset[index] for index in range(len(dataset))]
    if changed == "graph":
        graphs[0].x = tlx.convert_to_tensor([9, 2], dtype=tlx.int64)
    else:
        graphs[0].y = tlx.convert_to_tensor([[10.0]], dtype=tlx.float32)
    dataset.save_data(dataset.collate(graphs), dataset.processed_paths[0])

    second = trainer.load_or_prepare_preprocessing(args, spec)
    assert _cache_events(capsys) == ["created"]
    if changed == "graph":
        assert second[1]["train"][0]["serialized_token_ids"] != (
            first[1]["train"][0]["serialized_token_ids"])
        assert second[2] == first[2]
    else:
        assert (second[1]["train"][0]["labels"][0] * second[2]["std"][0]
                + second[2]["mean"][0]) == pytest.approx(10.0)
        assert second[2]["mean"] == pytest.approx([13.0 / 3.0])


def test_processed_split_change_rebuilds_training_statistics(trainer, tmp_path, capsys):
    root = tmp_path / "data"
    _write_preprocessing_raw(root)
    args = _preprocessing_args(root, tmp_path / "shared-cache")
    spec = trainer.resolve_dataset("aqsol")
    first = trainer.load_or_prepare_preprocessing(args, spec)
    assert _cache_events(capsys) == ["created"]
    split_path = root / "aqsol" / "processed" / "split_indices_v2.json"
    old_stat = split_path.stat()
    split_path.write_text(json.dumps({"train": [1, 2, 3], "val": [0], "test": [4]}),
                          encoding="utf-8")
    assert split_path.stat().st_size == old_stat.st_size
    os.utime(split_path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))

    second = trainer.load_or_prepare_preprocessing(args, spec)
    assert _cache_events(capsys) == ["created"]
    assert first[2]["mean"] == [1.0]
    assert second[2]["mean"] == [2.0]
    assert (second[1]["val"][0]["labels"][0] * second[2]["std"][0]
            + second[2]["mean"][0]) == pytest.approx(0.0)


def test_shared_cache_distinguishes_data_roots_by_content(trainer, tmp_path, capsys):
    cache = tmp_path / "shared-cache"
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    _write_preprocessing_raw(root_a)
    _write_preprocessing_raw(root_b, first_feature=9, first_label=10.0)
    spec = trainer.resolve_dataset("aqsol")
    first = trainer.load_or_prepare_preprocessing(_preprocessing_args(root_a, cache), spec)
    second = trainer.load_or_prepare_preprocessing(_preprocessing_args(root_b, cache), spec)

    assert _cache_events(capsys) == ["created", "created"]
    assert first[2]["mean"] == [1.0]
    assert second[2]["mean"] == pytest.approx([13.0 / 3.0])
    assert first[1]["train"][0]["serialized_token_ids"] != (
        second[1]["train"][0]["serialized_token_ids"])


def test_old_preprocessing_payload_rebuilds_current_data(trainer, tmp_path, capsys):
    root = tmp_path / "data"
    _write_preprocessing_raw(root)
    args = _preprocessing_args(root, tmp_path / "shared-cache")
    spec = trainer.resolve_dataset("aqsol")
    trainer.load_or_prepare_preprocessing(args, spec)
    assert _cache_events(capsys) == ["created"]
    path = trainer.preprocessing_cache_path(args, spec)
    with path.open("wb") as handle:
        pickle.dump({"preprocessing_cache_version": trainer.PREPROCESSING_CACHE_VERSION - 1,
                     "tokenizer": "old", "encoded": "old", "normalizer": "old",
                     "output_dim": "old"}, handle)

    rebuilt = trainer.load_or_prepare_preprocessing(args, spec)
    assert _cache_events(capsys) == ["created"]
    assert (rebuilt[1]["train"][0]["labels"][0] * rebuilt[2]["std"][0]
            + rebuilt[2]["mean"][0]) == pytest.approx(0.0)
    assert rebuilt[2]["mean"] == [1.0]


def test_preprocessing_cache_does_not_reuse_records_from_the_previous_schema(
        trainer, tmp_path, monkeypatch):
    args = SimpleNamespace(
        dataset="aqsol", encoder="bert", data_root=str(tmp_path / "data"),
        preprocessing_cache_dir=str(tmp_path / "cache"), bpe_merges=2,
        bpe_min_frequency=2, bpe_backend="python", num_serializations=1,
        max_length=64)
    spec = trainer.resolve_dataset("aqsol")
    splits = {"train": _aqsol_graphs(trainer, 3),
              "val": _aqsol_graphs(trainer, 1, offset=3),
              "test": _aqsol_graphs(trainer, 1, offset=4)}
    current_cache_version = trainer.PREPROCESSING_CACHE_VERSION
    assert current_cache_version > 4
    monkeypatch.setattr(trainer, "PREPROCESSING_CACHE_VERSION", current_cache_version - 1)
    stale_path = trainer.preprocessing_cache_path(args, spec)
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    with stale_path.open("wb") as handle:
        pickle.dump({"tokenizer": "stale", "encoded": "stale",
                     "normalizer": "stale", "output_dim": "stale"}, handle)
    monkeypatch.setattr(trainer, "PREPROCESSING_CACHE_VERSION", current_cache_version)
    monkeypatch.setattr(trainer, "load_dataset_splits", lambda *_: splits)

    tokenizer, encoded, _, _ = trainer.load_or_prepare_preprocessing(args, spec)

    assert tokenizer != "stale"
    assert encoded != "stale"
    assert trainer.preprocessing_cache_path(args, spec) != stale_path
    monkeypatch.setattr(trainer, "load_dataset_splits",
                        lambda *_: pytest.fail("new cache must be reused"))
    assert trainer.load_or_prepare_preprocessing(args, spec)[2]["mean"] == [1.0]


def test_legacy_finetune_checkpoint_is_rejected_before_loading_weights(trainer, tmp_path):
    model = torch.nn.Linear(1, 1)
    path = tmp_path / "old.pt"
    torch.save({"model": model.state_dict(), "epoch": 1, "score": 0.5}, path)

    with pytest.raises(ValueError, match="input protocol|schema"):
        trainer.restore_checkpoint(torch, path, model, torch.device("cpu"))


def test_current_cache_path_rejects_legacy_payload(trainer, tmp_path, monkeypatch):
    args = SimpleNamespace(
        dataset="aqsol", encoder="bert", data_root=str(tmp_path / "data"),
        preprocessing_cache_dir=str(tmp_path / "cache"), bpe_merges=2,
        bpe_min_frequency=2, bpe_backend="python", num_serializations=1,
        max_length=64)
    spec = trainer.resolve_dataset("aqsol")
    path = trainer.preprocessing_cache_path(args, spec)
    path.parent.mkdir(parents=True)
    with path.open("wb") as handle:
        pickle.dump({"tokenizer": "old", "encoded": "old",
                     "normalizer": "old", "output_dim": "old"}, handle)
    splits = {"train": _aqsol_graphs(trainer, 3),
              "val": _aqsol_graphs(trainer, 1, offset=3),
              "test": _aqsol_graphs(trainer, 1, offset=4)}
    monkeypatch.setattr(trainer, "load_dataset_splits", lambda *_: splits)

    tokenizer, encoded, _, _ = trainer.load_or_prepare_preprocessing(args, spec)

    assert tokenizer != "old"
    assert encoded != "old"
    with path.open("rb") as handle:
        assert pickle.load(handle)["preprocessing_cache_version"] == (
            trainer.PREPROCESSING_CACHE_VERSION)


def test_warmup_cosine_scheduler_warms_up_then_decays(trainer):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = trainer.build_warmup_cosine_scheduler(
        torch, optimizer, total_steps=100, warmup_ratio=0.12)
    multiplier = scheduler.lr_lambdas[0]

    assert multiplier(0) == pytest.approx(1 / 12)
    assert multiplier(11) == pytest.approx(1.0)
    assert multiplier(50) < 1.0
    assert multiplier(100) == pytest.approx(0.01)


def test_phase_training_uses_phase_specific_gradient_clipping(trainer, monkeypatch):
    class CountingScheduler:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    class TinyMLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids, attention, task):
            return self.scale * torch.ones((len(input_ids), input_ids.shape[1], 7))

    class TinySupervised(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, input_ids, attention, task):
            return self.weight.expand(len(input_ids), 1)

    clipped = []
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_",
                        lambda _parameters, max_norm: clipped.append(max_norm))
    tokenizer = SimpleNamespace(special_tokens=SimpleNamespace(
        pad_token_id=0, cls_token_id=3, sep_token_id=4,
        component_sep_token_id=5, mask_token_id=6), vocabulary={7: 7})
    loader = [(torch.tensor([[3, 2, 4]]), torch.ones((1, 3), dtype=torch.long),
               torch.tensor([[1.0]]), torch.tensor([0]))]
    mlm_model = TinyMLM()
    mlm_scheduler = CountingScheduler()
    trainer.train_mlm_epoch(
        torch, mlm_model, loader, torch.optim.SGD(mlm_model.parameters(), lr=0.1),
        mlm_scheduler, tokenizer, SimpleNamespace(mask_probability=1.0,
                                                   pretrain_max_grad_norm=2.0),
        torch.device("cpu"), seed=7)
    supervised_model = TinySupervised()
    supervised_scheduler = CountingScheduler()
    trainer.train_downstream_epoch(
        torch, supervised_model, loader,
        torch.optim.SGD(supervised_model.parameters(), lr=0.1), supervised_scheduler,
        trainer.resolve_dataset("aqsol"), SimpleNamespace(
            finetune_max_grad_norm=0.5, finetune_noise_probability=0.0,
            finetune_noise_std=0.01),
        torch.device("cpu"))

    assert clipped == [2.0, 0.5]
    assert mlm_scheduler.steps == 1
    assert supervised_scheduler.steps == 1


def test_phase_training_accumulates_gradients_before_optimizer_step(trainer, monkeypatch):
    class CountingScheduler:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    class TinyMLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids, attention, task):
            return self.scale * torch.ones((len(input_ids), input_ids.shape[1], 7))

    tokenizer = SimpleNamespace(special_tokens=SimpleNamespace(
        pad_token_id=0, cls_token_id=3, sep_token_id=4,
        component_sep_token_id=5, mask_token_id=6), vocabulary={7: 7})
    loader = [(torch.tensor([[3, 2, 4]]), torch.ones((1, 3), dtype=torch.long),
               torch.tensor([[1.0]]), torch.tensor([0])) for _ in range(3)]
    model = TinyMLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer_steps = []
    original_step = optimizer.step
    monkeypatch.setattr(optimizer, "step", lambda: (optimizer_steps.append(1), original_step())[1])
    scheduler = CountingScheduler()

    trainer.train_mlm_epoch(
        torch, model, loader, optimizer, scheduler, tokenizer,
        SimpleNamespace(mask_probability=1.0, pretrain_max_grad_norm=2.0,
                        gradient_accumulation_steps=2), torch.device("cpu"), seed=7)

    assert len(optimizer_steps) == 2
    assert scheduler.steps == 2


@pytest.mark.parametrize(
    ("batch_count", "accumulation_steps", "expected"),
    [
        (6, 4, [4, 4, 4, 4, 2, 2]),
        (8, 4, [4, 4, 4, 4, 4, 4, 4, 4]),
        (3, 1, [1, 1, 1]),
        (3, 4, [3, 3, 3]),
    ],
)
def test_gradient_accumulation_window_size_uses_the_full_window(
        trainer, batch_count, accumulation_steps, expected):
    assert [
        trainer.gradient_accumulation_window_size(
            batch_index, batch_count, accumulation_steps)
        for batch_index in range(batch_count)
    ] == expected


def _accumulated_gradients(trainer, monkeypatch, path, batch_gradients, accumulation_steps):
    class CountingScheduler:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    class ScalarModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

    model = ScalarModel()
    loader = [
        (torch.tensor([[gradient]]), torch.ones((1, 1), dtype=torch.long),
         torch.tensor([[0.0]]), torch.tensor([batch_index]))
        for batch_index, gradient in enumerate(batch_gradients)
    ]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    gradients = []
    original_step = optimizer.step

    def capture_gradient_then_step():
        gradients.append(float(model.scale.grad))
        original_step()

    monkeypatch.setattr(optimizer, "step", capture_gradient_then_step)
    scheduler = CountingScheduler()
    if path == "mlm":
        tokenizer = SimpleNamespace(special_tokens=SimpleNamespace(
            pad_token_id=0, cls_token_id=3, sep_token_id=4,
            component_sep_token_id=5, mask_token_id=6), vocabulary={7: 7})

        def known_mlm_loss(_torch, _model, input_ids, *_args):
            return model.scale * input_ids[0, 0].float()

        monkeypatch.setattr(trainer, "mlm_loss", known_mlm_loss)
        epoch_loss = trainer.train_mlm_epoch(
            torch, model, loader, optimizer, scheduler, tokenizer,
            SimpleNamespace(mask_probability=1.0, pretrain_max_grad_norm=1e6,
                            gradient_accumulation_steps=accumulation_steps),
            torch.device("cpu"), seed=7)
    else:
        def known_logits(_torch, _model, input_ids, *_args):
            return model.scale * input_ids[:, 0].float()

        monkeypatch.setattr(trainer, "supervised_logits", known_logits)
        monkeypatch.setattr(trainer, "supervised_loss", lambda _torch, logits, *_args: logits.mean())
        epoch_loss = trainer.train_downstream_epoch(
            torch, model, loader, optimizer, scheduler, trainer.resolve_dataset("aqsol"),
            SimpleNamespace(finetune_max_grad_norm=1e6, finetune_noise_probability=0.0,
                            finetune_noise_std=0.0,
                            gradient_accumulation_steps=accumulation_steps),
            torch.device("cpu"))
    return gradients, scheduler.steps, epoch_loss


@pytest.mark.parametrize("path", ("mlm", "downstream"))
def test_training_accumulation_averages_each_partial_window(
        trainer, monkeypatch, path):
    gradients, scheduler_steps, epoch_loss = _accumulated_gradients(
        trainer, monkeypatch, path, [1, 2, 3, 4, 5, 6], accumulation_steps=4)

    assert gradients == pytest.approx([2.5, 5.5])
    assert scheduler_steps == 2
    assert epoch_loss == pytest.approx(3.5)


@pytest.mark.parametrize(
    ("batch_gradients", "accumulation_steps", "expected_gradients"),
    [
        ([1, 2, 3, 4, 5, 6, 7, 8], 4, [2.5, 6.5]),
        ([1, 2, 3], 1, [1.0, 2.0, 3.0]),
        ([1, 2, 3], 4, [2.0]),
    ],
)
@pytest.mark.parametrize("path", ("mlm", "downstream"))
def test_training_accumulation_matches_complete_and_short_windows(
        trainer, monkeypatch, path, batch_gradients, accumulation_steps, expected_gradients):
    gradients, scheduler_steps, _ = _accumulated_gradients(
        trainer, monkeypatch, path, batch_gradients, accumulation_steps)

    assert gradients == pytest.approx(expected_gradients)
    assert scheduler_steps == len(expected_gradients)


def test_set_seed_controls_python_numpy_torch_cuda_and_cudnn(trainer, monkeypatch):
    calls = []
    fake_numpy = SimpleNamespace(random=SimpleNamespace(
        seed=lambda value: calls.append(("numpy", value))))
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setattr(trainer.random, "seed", lambda value: calls.append(("python", value)))

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def manual_seed_all(value):
                calls.append(("cuda", value))

        class backends:
            class cudnn:
                deterministic = False
                benchmark = True

        @staticmethod
        def manual_seed(value):
            calls.append(("torch", value))

    trainer.set_seed(FakeTorch, 17)

    assert calls == [("python", 17), ("numpy", 17), ("torch", 17), ("cuda", 17)]
    assert FakeTorch.backends.cudnn.deterministic is True
    assert FakeTorch.backends.cudnn.benchmark is False


def test_mlm_corruption_uses_80_10_10_and_excludes_special_tokens(trainer, monkeypatch):
    tokenizer = SimpleNamespace(
        special_tokens=SimpleNamespace(
            pad_token_id=0, unk_token_id=1, mask_token_id=2, cls_token_id=3,
            sep_token_id=4, node_start_token_id=5, node_end_token_id=6,
            component_sep_token_id=7),
        vocabulary={100: 8, 101: 9, 102: 10, 103: 11})
    input_ids = torch.tensor([[0, 3, 8, 9, 10, 4, 2, 5, 6, 7]])
    attention = torch.ones_like(input_ids)
    draws = iter((
        torch.zeros_like(input_ids, dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 0.1, 0.85, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    ))
    monkeypatch.setattr(torch, "rand",
                        lambda _shape, generator=None, device=None: next(draws).to(device))
    monkeypatch.setattr(
        torch, "randint",
        lambda _high, shape, generator=None, device=None:
        torch.zeros(shape, dtype=torch.long, device=device))

    corrupted, labels = trainer.mask_for_mlm(
        torch, input_ids, attention, tokenizer, probability=0.09,
        generator=torch.Generator().manual_seed(7))

    assert labels.tolist() == [[-100, -100, 8, 9, 10, -100, -100, -100, -100, -100]]
    assert corrupted.tolist() == [[0, 3, 2, 8, 10, 4, 2, 5, 6, 7]]
    assert corrupted[0, 3].item() not in set(vars(tokenizer.special_tokens).values())


def test_random_swap_uses_local_window_and_preserves_special_positions(trainer):
    tokens = [7, *range(8, 28)]
    unchanged = trainer.random_swap_serialized_tokens(
        tokens, {7}, probability=0.0, ratio=0.5, window=3, rng=random.Random(3))
    swapped = trainer.random_swap_serialized_tokens(
        tokens, {7}, probability=1.0, ratio=0.5, window=3, rng=random.Random(3))

    assert unchanged == tokens
    assert swapped != tokens
    assert swapped[0] == 7
    assert sorted(swapped) == sorted(tokens)


def test_sequence_mask_only_changes_configured_non_special_positions(trainer):
    tokens = [7, *range(8, 28)]
    masked = trainer.sequence_mask_serialized_tokens(
        tokens, {7}, mask_token_id=2, probability=1.0, ratio=0.1,
        rng=random.Random(5))

    assert masked[0] == 7
    assert masked.count(2) == 2
    assert all(masked[index] == tokens[index] for index in range(1, len(tokens))
               if masked[index] != 2)


def test_augmented_overlength_sequence_falls_back_without_truncation(trainer):
    tokenizer = SimpleNamespace(
        special_tokens=SimpleNamespace(mask_token_id=2),
        encode_tokens=lambda tokens: [3, *tokens, 4])
    original = [3, 8, 9, 4]
    fallback = trainer.augment_record_input_ids(
        tokenizer, original, list(range(8, 18)), {7},
        {"swap_probability": 1.0, "swap_ratio": 0.5, "swap_window": 3},
        max_length=4, rng=random.Random(2))

    assert fallback == original


def test_validation_and_test_augmentation_are_disabled(trainer):
    tokens = [7, *range(8, 28)]

    assert trainer.augment_serialized_tokens(tokens, {7}, None, random.Random(9)) == tokens


def test_gaussian_noise_only_changes_training_pooled_representation(trainer):
    class PooledModel:
        def __call__(self, input_ids, attention, task):
            if task == "pooled":
                return torch.ones((len(input_ids), 2))
            return torch.ones((len(input_ids), 2))

        @staticmethod
        def _task_logits(pooled):
            return pooled

    model = PooledModel()
    input_ids = torch.tensor([[3, 8, 4]])
    attention = torch.ones_like(input_ids)
    noisy = trainer.supervised_logits(
        torch, model, input_ids, attention, noise_probability=1.0, noise_std=0.01,
        generator=torch.Generator().manual_seed(4))
    clean = trainer.supervised_logits(
        torch, model, input_ids, attention, noise_probability=0.0, noise_std=0.01,
        generator=torch.Generator().manual_seed(4))

    assert not torch.equal(noisy, clean)
    assert torch.equal(clean, torch.ones((1, 2)))


def test_main_runs_requested_single_experiment(trainer, monkeypatch, tmp_path):
    calls = []

    def run_single(args, config, seed, run_index):
        calls.append((args, config, seed, run_index))
        return _run_result(seed, run_index, 1.0)

    monkeypatch.setattr(trainer, "run_single_experiment", run_single)

    summary = trainer.main(["--smoke", "--dataset", "aqsol", "--encoder", "bert",
                            "--seeds", "42", "--output-dir", str(tmp_path)])
    assert len(calls) == 1
    assert calls[0][1] == trainer.PAPER_CONFIGS[("aqsol", "bert")]
    assert calls[0][2] == 42
    assert calls[0][3] == 0
    assert summary["seeds"] == [42]
    assert summary["run_metrics"] == [1.0]
    assert summary["complete"] is True


def test_main_orchestrates_five_runs_and_isolates_summaries(trainer, monkeypatch, tmp_path):
    calls = []

    def run_single(_args, _config, seed, run_index):
        calls.append((seed, run_index))
        return _run_result(seed, run_index, float(run_index + 1))

    monkeypatch.setattr(trainer, "run_single_experiment", run_single)

    summary = trainer.main(["--dataset", "aqsol", "--encoder", "bert",
                            "--output-dir", str(tmp_path)])

    assert calls == [(42, 0), (43, 1), (44, 2), (45, 3), (46, 4)]
    assert summary["run_metrics"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    for seed, run_index in calls:
        assert (tmp_path / "aqsol" / "bert" /
                f"run_{run_index + 1:02d}_seed_{seed}" / "summary.json").is_file()


def test_aggregate_run_results_reports_sample_standard_deviation(trainer):
    metrics = [1.0, 2.0, 3.0, 4.0, 5.0]
    summary = trainer.aggregate_run_results([
        _run_result(42, 0, metrics[0]), _run_result(43, 1, metrics[1]),
        _run_result(44, 2, metrics[2]), _run_result(45, 3, metrics[3]),
        _run_result(46, 4, metrics[4]),
    ])

    assert summary["run_metrics"] == metrics
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["std"] == pytest.approx(math.sqrt(2.5))
    assert summary["std_ddof"] == 1
    assert summary["display"] == f"{summary['mean']} ± {summary['std']}"
    assert summary["complete"] is True


def test_aggregate_run_results_uses_zero_std_for_one_metric(trainer):
    summary = trainer.aggregate_run_results([_run_result(42, 0, 1.0)])

    assert summary["mean"] == 1.0
    assert summary["std"] == 0.0
    assert summary["std_ddof"] == 1


def test_main_does_not_write_complete_summary_when_a_run_fails(trainer, monkeypatch, tmp_path):
    def run_single(_args, _config, seed, run_index):
        if seed == 44:
            raise RuntimeError("run failed")
        return _run_result(seed, run_index, float(run_index + 1))

    monkeypatch.setattr(trainer, "run_single_experiment", run_single)

    with pytest.raises(RuntimeError, match="run failed"):
        trainer.main(["--dataset", "aqsol", "--encoder", "bert", "--seeds", "42", "43", "44",
                      "--output-dir", str(tmp_path)])

    partial = json.loads((tmp_path / "aqsol" / "bert" / "summary.json").read_text())
    assert partial["complete"] is False
    assert partial["run_metrics"] == [1.0, 2.0]
    assert "mean" not in partial
    assert "std" not in partial
    assert "display" not in partial


def test_tiny_pipeline_runs_full_train_mlm_then_finetune_and_final_test(trainer, monkeypatch, tmp_path):
    splits = {
        "train": _aqsol_graphs(trainer, 8),
        "val": _aqsol_graphs(trainer, 4, 20),
        "test": _aqsol_graphs(trainer, 4, 40),
    }
    monkeypatch.setattr(trainer, "synthetic_splits", lambda _spec: splits)

    summary = trainer.main([
        "--smoke", "--dataset", "aqsol", "--encoder", "bert", "--device", "cpu",
        "--seeds", "42",
        "--output-dir", str(tmp_path),
    ])

    run_directory = tmp_path / "aqsol" / "bert" / "run_01_seed_42"
    run_summary = json.loads((run_directory / "summary.json").read_text())
    assert Path(run_summary["checkpoints"]["finetune"]).is_file()
    assert run_summary["mlm"]["epochs_completed"] == 2
    assert all(math.isfinite(row["train_loss"]) for row in run_summary["mlm"]["history"])
    assert all("val_loss" not in row for row in run_summary["mlm"]["history"])
    assert all(math.isfinite(row["train_loss"]) and math.isfinite(row["validation"]["metric"])
               for row in run_summary["finetune"]["history"])
    assert math.isfinite(run_summary["finetune"]["test"]["metric"])
    assert summary["run_metrics"] == [run_summary["metric"]]


def test_tiny_pipeline_writes_summary_file(trainer, tmp_path):
    summary = trainer.main([
        "--smoke", "--dataset", "aqsol", "--encoder", "bert", "--device", "cpu",
        "--seeds", "42",
        "--output-dir", str(tmp_path),
    ])

    saved = tmp_path / "aqsol" / "bert" / "summary.json"
    assert saved.is_file()
    assert saved.read_text() == json.dumps(summary, indent=2, allow_nan=True)


def test_tiny_pipeline_emits_preprocessing_and_training_progress(trainer, tmp_path, capsys):
    trainer.main([
        "--smoke", "--dataset", "aqsol", "--encoder", "bert", "--device", "cpu",
        "--seeds", "42",
        "--output-dir", str(tmp_path),
    ])

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()
              if line.startswith('{"event"')]

    assert {"event": "stage", "stage": "tokenizer_fit", "status": "started"} in events
    assert any(event.get("phase") == "pretrain" for event in events)
    assert any(event.get("phase") == "finetune" for event in events)


def test_tiny_pretrained_gte_smoke_requires_explicit_random_initialization(trainer, tmp_path):
    with pytest.raises(ValueError, match="tiny smoke architecture.*allow-random-gte-init"):
        trainer.main([
            "--smoke", "--dataset", "aqsol", "--encoder", "gte", "--device", "cpu",
            "--seeds", "42", "--output-dir", str(tmp_path),
        ])


def test_tiny_random_gte_smoke_runs_only_when_explicitly_requested(trainer, tmp_path):
    summary = trainer.main([
        "--smoke", "--dataset", "aqsol", "--encoder", "gte", "--device", "cpu",
        "--allow-random-gte-init", "--seeds", "42", "--output-dir", str(tmp_path),
    ])

    assert summary["encoder"] == "gte"
    assert summary["complete"] is True


def test_non_smoke_gte_does_not_require_random_initialization(trainer, tmp_path, monkeypatch):
    monkeypatch.setattr(
        trainer, "run_single_experiment",
        lambda _args, _config, seed, run_index=0: {
            "dataset": "aqsol", "encoder": "gte", "seed": seed,
            "run_index": run_index, "metric_name": "MAE", "metric": 0.1,
        },
    )

    summary = trainer.main([
        "--dataset", "aqsol", "--encoder", "gte", "--seeds", "42",
        "--output-dir", str(tmp_path),
    ])

    assert summary["encoder"] == "gte"
    assert summary["complete"] is True


def test_formal_gte_preset_uses_the_official_loader(trainer, monkeypatch):
    from gammagl.models.graph_gte import GraphGTE

    args = trainer.build_parser().parse_args(["--dataset", "aqsol", "--encoder", "gte"])
    args = trainer.apply_preset(args)
    calls, loaded = {}, SimpleNamespace(pretrained_manifest={
        "pretrained": True, "reproduction": True, "encoder_coverage": 1.0,
        "missing": [], "unexpected": [], "shape_mismatches": [],
    })

    def from_pretrained(_cls, **kwargs):
        calls.update(kwargs)
        return loaded

    monkeypatch.setattr(GraphGTE, "from_pretrained", classmethod(from_pretrained))

    assert trainer.make_model(args, vocab_size=31, pad_token_id=0, output_dim=1) is loaded
    assert calls["vocab_size"] == 31
    assert calls["max_position_embeddings"] == 8192
    assert trainer.gte_initialization_summary(loaded, args)["reproduction"] is True


def test_formal_gte_tokenizer_uses_the_official_padding_id(trainer):
    args = trainer.apply_preset(trainer.build_parser().parse_args([
        "--dataset", "aqsol", "--encoder", "gte",
    ]))

    tokenizer = trainer.make_tokenizer(args, _aqsol_graphs(trainer, 2))

    assert tokenizer.special_tokens.pad_token_id == 1
    assert tokenizer.special_tokens.unk_token_id == 0


def test_formal_gte_loader_failure_is_not_replaced_with_random_initialization(trainer, monkeypatch):
    from gammagl.models.graph_gte import GraphGTE

    args = trainer.apply_preset(trainer.build_parser().parse_args([
        "--dataset", "aqsol", "--encoder", "gte",
    ]))

    def rejected_loader(_cls, **_kwargs):
        raise RuntimeError("GTE checkpoint SHA-256 mismatch")

    monkeypatch.setattr(GraphGTE, "from_pretrained", classmethod(rejected_loader))

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        trainer.make_model(args, vocab_size=31, pad_token_id=0, output_dim=1)


def test_explicit_random_gte_development_mode_is_not_a_reproduction(trainer, monkeypatch):
    from gammagl.models.graph_gte import GraphGTE

    args = trainer.apply_preset(trainer.build_parser().parse_args([
        "--dataset", "aqsol", "--encoder", "gte", "--allow-random-gte-init",
        "--hidden-size", "16", "--num-hidden-layers", "1",
        "--num-attention-heads", "4", "--intermediate-size", "32",
    ]))
    monkeypatch.setattr(GraphGTE, "from_pretrained", classmethod(
        lambda *_args, **_kwargs: pytest.fail("random development mode must not load a checkpoint")))

    model = trainer.make_model(args, vocab_size=31, pad_token_id=0, output_dim=1)

    assert isinstance(model, GraphGTE)
    assert trainer.gte_initialization_summary(model, args) == {
        "pretrained": False, "reproduction": False, "mode": "random_development",
    }


def test_train_only_bpe_and_normalization_do_not_use_validation_or_test(trainer):
    train = _aqsol_graphs(trainer, 2)
    held_out = trainer.GraphRecord([[0], [1]], [101, 102], [5], [1000.0])
    args = SimpleNamespace(bpe_merges=2, bpe_min_frequency=2, bpe_backend="python",
                           num_serializations=1)
    tokenizer = trainer.make_tokenizer(args, train)
    before = (dict(tokenizer.vocabulary), list(tokenizer.bpe.codebook.merge_rules))

    held_out_encoding = tokenizer.encode_graph(held_out)
    encoded = {
        "train": [_record(0, [1.0]), _record(1, [3.0])],
        "val": [_record(0, [1000.0])],
        "test": [_record(0, [-1000.0])],
    }
    normalizer, output_dim = trainer.normalize_labels(encoded, trainer.resolve_dataset("aqsol"))

    assert tokenizer.special_tokens.unk_token_id in held_out_encoding.input_ids
    assert (tokenizer.vocabulary, tokenizer.bpe.codebook.merge_rules) == before
    assert output_dim == 1
    assert normalizer["mean"] == [2.0]
    assert normalizer["std"] == [1.0]


def test_aqsol_metric_inverse_normalizes_and_uses_raw_mae(trainer):
    normalizer = {"mean": [10.0], "std": [2.0]}
    loader = trainer.make_loader(torch, [_record(0, [0.0])], 1, 0, False, False)

    result = trainer.evaluate_downstream(
        torch, FixedLogits([[1.0]]), loader, trainer.resolve_dataset("aqsol"), normalizer,
        torch.device("cpu"))

    assert result["metric_space"] == "raw_label"
    assert result["mae"] == pytest.approx(2.0)


def test_mutagenicity_metric_averages_serializations_before_accuracy(trainer):
    logits = [[3.0, 1.0], [2.0, 0.0], [0.0, 2.0], [1.0, 4.0]]
    records = [_record(0, [0.0]), _record(0, [0.0]), _record(1, [1.0]), _record(1, [1.0])]
    loader = trainer.make_loader(torch, records, 4, 0, False, False)

    result = trainer.evaluate_downstream(
        torch, FixedLogits(logits), loader, trainer.resolve_dataset("mutagenicity"), None,
        torch.device("cpu"))

    assert result["num_graphs"] == 2
    assert result["metric_space"] == "class_label"
    assert result["accuracy"] == pytest.approx(1.0)


def test_dblp_metric_uses_class_logits_and_accuracy(trainer):
    loader = trainer.make_loader(torch, [_record(0, [1.0])], 1, 0, False, False)

    result = trainer.evaluate_downstream(
        torch, FixedLogits([[0.2, 0.8]]), loader,
        trainer.resolve_dataset("dblp"), None, torch.device("cpu"))

    assert result["accuracy"] == pytest.approx(1.0)
    assert result["metric_space"] == "class_label"


def test_mlm_runs_configured_epochs_without_validation_or_checkpoint_restore(trainer, monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    train_epochs = []
    scheduler_arguments = []
    args = SimpleNamespace(pretrain_learning_rate=1e-3, weight_decay=0.0,
                           pretrain_epochs=5, pretrain_warmup_ratio=0.12, seed=7)

    def train(*_args):
        train_epochs.append(len(train_epochs) + 1)
        model.weight.data.fill_(train_epochs[-1])
        return 0.1

    monkeypatch.setattr(trainer, "train_mlm_epoch", train)
    monkeypatch.setattr(
        trainer, "build_warmup_cosine_scheduler",
        lambda _torch, _optimizer, total_steps, warmup_ratio:
        scheduler_arguments.append((total_steps, warmup_ratio)) or object())
    result = trainer.run_mlm(torch, model, [None], None, args, torch.device("cpu"))

    assert train_epochs == [1, 2, 3, 4, 5]
    assert result["epochs_completed"] == 5
    assert result["final_loss"] == 0.1
    assert all(set(row) == {"epoch", "train_loss"} for row in result["history"])
    assert scheduler_arguments == [(5, 0.12)]


def test_single_run_uses_every_downstream_training_graph_for_mlm(trainer, monkeypatch, tmp_path):
    splits = {
        "train": _aqsol_graphs(trainer, 4),
        "val": _aqsol_graphs(trainer, 2, 20),
        "test": _aqsol_graphs(trainer, 2, 40),
    }
    observed = {}
    monkeypatch.setattr(trainer, "synthetic_splits", lambda _spec: splits)

    def run_mlm(_torch, _model, train_loader, _tokenizer, _args, _device):
        observed["graph_ids"] = {group[0]["graph_id"] for group in train_loader.dataset}
        return {"epochs_completed": 2, "final_loss": 0.1,
                "history": [{"epoch": 1, "train_loss": 0.1},
                            {"epoch": 2, "train_loss": 0.1}]}

    monkeypatch.setattr(trainer, "run_mlm", run_mlm)
    monkeypatch.setattr(
        trainer, "run_finetuning",
        lambda *_args: {"best_epoch": 1, "best_validation_metric": 0.1,
                        "history": [], "test": {"metric": 0.1}})

    trainer.main([
        "--smoke", "--dataset", "aqsol", "--encoder", "bert", "--device", "cpu",
        "--seeds", "42", "--output-dir", str(tmp_path),
    ])

    assert observed["graph_ids"] == {0, 1, 2, 3}


def test_finetune_early_stopping_restores_best_before_test(trainer, monkeypatch, tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    epochs, validation_metrics, validation_calls, test_weights = [], iter((2.0, 1.0, 3.0)), [], []
    scheduler_arguments = []
    args = SimpleNamespace(learning_rate=1e-3, weight_decay=0.0, finetune_epochs=5,
                           finetune_warmup_ratio=0.025, early_stopping_patience=1, seed=7)

    def train(*_args):
        epochs.append(len(epochs) + 1)
        model.weight.data.fill_(epochs[-1])
        return 0.1

    def evaluate(*_args):
        if len(validation_calls) < 3:
            validation_calls.append(True)
            return {"loss": 0.1, "metric": next(validation_metrics)}
        test_weights.append(float(model.weight.detach()))
        return {"loss": 0.1, "metric": test_weights[-1]}

    monkeypatch.setattr(trainer, "train_downstream_epoch", train)
    monkeypatch.setattr(trainer, "evaluate_downstream", evaluate)
    monkeypatch.setattr(
        trainer, "build_warmup_cosine_scheduler",
        lambda _torch, _optimizer, total_steps, warmup_ratio:
        scheduler_arguments.append((total_steps, warmup_ratio)) or object())
    result = trainer.run_finetuning(
        torch, model, [None], None, None, trainer.resolve_dataset("aqsol"), None, args,
        torch.device("cpu"), tmp_path / "finetune.pt")

    assert result["best_epoch"] == 2
    assert len(result["history"]) == 3
    assert result["test"]["metric"] == 2.0
    assert test_weights == [2.0]
    assert scheduler_arguments == [(5, 0.025)]
