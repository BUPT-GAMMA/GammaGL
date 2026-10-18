import hashlib
import importlib.util
import io
import json
import os
import pickle
import shutil
import tarfile
from pathlib import Path

import pytest
import tensorlayerx as tlx

import gammagl.datasets._graph_tokenizer_download as graph_tokenizer_download
from gammagl.datasets import AQSOL, GraphTokenizerDBLP, Mutagenicity
from gammagl.data import Graph
from gammagl.transforms.graph_serializer import FrequencyGuidedEulerianSerializer


class FakeDGLGraph:
    """Minimal pickleable DGL-shaped graph for released bundle fixtures."""

    def __init__(self, src, dst, ndata, edata):
        self._src = src
        self._dst = dst
        self.ndata = ndata
        self.edata = edata

    def edges(self):
        return self._src, self._dst


class ExplicitDirectedDGLGraph(FakeDGLGraph):
    def is_directed(self):
        return True


def _graph(ndata, edata):
    return FakeDGLGraph([0], [1], ndata, edata)


def _write_raw(root, name, samples, splits):
    raw_dir = root / name / "raw"
    raw_dir.mkdir(parents=True)
    with (raw_dir / "data.pkl").open("wb") as handle:
        pickle.dump(samples, handle)
    for split, indices in splits.items():
        (raw_dir / f"{split}_index.json").write_text(
            json.dumps(indices), encoding="utf-8")


def _download_module():
    path = Path(__file__).resolve().parents[2] / "gammagl" / "datasets" / "_graph_tokenizer_download.py"
    spec = importlib.util.spec_from_file_location("graph_tokenizer_download_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_bundle(path, datasets):
    with tarfile.open(path, "w:gz") as archive:
        for name, samples in datasets.items():
            files = {
                f"release/{name}/data.pkl": pickle.dumps(samples),
                f"release/{name}/train_index.json": b"[0]",
                f"release/{name}/val_index.json": b"[]",
                f"release/{name}/test_index.json": b"[]",
            }
            for member, content in files.items():
                info = tarfile.TarInfo(member)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))


def test_mutagenicity_loads_official_split_and_default_edge_token(tmp_path):
    _write_raw(tmp_path, "mutagenicity", [
        (_graph({"node_token_ids": [[5], [9]]}, {}), 1),
    ], {"train": [0], "val": [], "test": []})

    dataset = Mutagenicity(root=str(tmp_path))
    graph = dataset[0]

    assert dataset.get_idx_split() == {"train": [0], "val": [], "test": []}
    assert dataset.num_classes == 2
    assert tlx.convert_to_numpy(graph.x).tolist() == [5, 9]
    assert tlx.convert_to_numpy(graph.edge_attr).tolist() == [0]
    assert tlx.convert_to_numpy(graph.y).shape == (1, 1)
    assert tlx.convert_to_numpy(graph.y).dtype.kind == "f"


def test_graph_tokenizer_dblp_loads_official_split_and_default_edge_token(tmp_path):
    _write_raw(tmp_path, "dblp", [
        (_graph({"node_token_ids": [[7], [11]]}, {}), 0),
    ], {"train": [0], "val": [], "test": []})

    dataset = GraphTokenizerDBLP(root=str(tmp_path))
    graph = dataset[0]

    assert dataset.get_idx_split() == {"train": [0], "val": [], "test": []}
    assert dataset.num_classes == 2
    assert tlx.convert_to_numpy(graph.x).tolist() == [7, 11]
    assert tlx.convert_to_numpy(graph.edge_attr).tolist() == [0]
    assert float(tlx.convert_to_numpy(graph.y)[0, 0]) == 0.0


@pytest.mark.parametrize("dataset_class,name,labels", [
    (Mutagenicity, "mutagenicity", [0., 1.]),
    (GraphTokenizerDBLP, "dblp", [1., 0.]),
    (AQSOL, "aqsol", [0.37, -1.42]),
])
def test_processed_reload_preserves_integer_tokens_and_float_targets(tmp_path, dataset_class, name, labels):
    # This ID also detects an intermediate float32 conversion / rounding.
    token_id = 2 ** 25 + 1
    raw_graph = {"edge_index": [[0, 1], [1, 0]],
                 "node_token_ids": [token_id, 9], "edge_token_ids": [2, 2]}
    _write_raw(tmp_path, name, [(raw_graph, label) for label in labels],
               {"train": [0], "val": [1], "test": []})
    for dataset in [dataset_class(root=str(tmp_path)), dataset_class(root=str(tmp_path))]:
        for index, label in enumerate(labels):
            graph = dataset[index]
            for field in [graph.x, graph.edge_attr, graph.edge_index]:
                assert tlx.convert_to_numpy(field).dtype.kind == "i"
            assert tlx.convert_to_numpy(graph.x).tolist() == [token_id, 9]
            target = tlx.convert_to_numpy(graph.y)
            assert target.dtype.kind == "f"
            assert float(target[0, 0]) == pytest.approx(label)


@pytest.mark.parametrize("graph_data", [
    {"edge_index": [[0], [1]], "node_token_ids": [5, 9], "directed": True},
    Graph(edge_index=[[0], [1]], x=[5, 9], directed=True),
    Graph(edge_index=[[0], [1]], x=[5, 9], is_directed=True),
    ExplicitDirectedDGLGraph([0], [1], {"node_token_ids": [[5], [9]]}, {}),
])
def test_adapter_rejects_explicit_directed_semantics(tmp_path, graph_data):
    _write_raw(tmp_path, "mutagenicity", [(graph_data, 1)],
               {"train": [0], "val": [], "test": []})
    with pytest.raises(ValueError, match="directed graph semantics"):
        Mutagenicity(root=str(tmp_path))


@pytest.mark.parametrize("edges,directed", [
    ([[0], [1]], False),
    ([[0], [1]], None),
    ([[0, 1], [1, 0]], None),
])
def test_adapter_accepts_undirected_coo_storage(tmp_path, edges, directed):
    graph_data = {"edge_index": edges, "node_token_ids": [5, 9]}
    if directed is not None:
        graph_data["directed"] = directed
    _write_raw(tmp_path, "mutagenicity", [(graph_data, 1)],
               {"train": [0], "val": [], "test": []})
    graph = Mutagenicity(root=str(tmp_path))[0]
    assert FrequencyGuidedEulerianSerializer().serialize(graph).token_ids


def test_adapter_does_not_reuse_old_processed_graphs(tmp_path):
    raw = {"edge_index": [[0], [1]], "node_token_ids": [5, 9]}
    _write_raw(tmp_path, "mutagenicity", [(raw, 1)],
               {"train": [0], "val": [], "test": []})
    first = Mutagenicity(root=str(tmp_path))
    processed = Path(first.processed_dir)
    (processed / (tlx.BACKEND + "_data.pt")).write_bytes(
        Path(first.processed_paths[0]).read_bytes())
    (processed / "split_indices.json").write_bytes(
        Path(first.processed_paths[1]).read_bytes())
    raw["directed"] = True
    with (Path(first.raw_dir) / "data.pkl").open("wb") as handle:
        pickle.dump([(raw, 1)], handle)
    for path in first.processed_paths:
        Path(path).unlink()
    with pytest.raises(ValueError, match="directed graph semantics"):
        Mutagenicity(root=str(tmp_path))


@pytest.mark.parametrize("dataset_class,dataset_dir", [
    (Mutagenicity, "mutagenicity"),
    (GraphTokenizerDBLP, "dblp"),
])
@pytest.mark.parametrize("label", [0, 1, 0.0, 1.0])
def test_binary_classification_labels_accept_only_valid_classes(
        tmp_path, dataset_class, dataset_dir, label):
    _write_raw(tmp_path, dataset_dir, [
        (_graph({"node_token_ids": [[5], [9]]}, {}), label),
    ], {"train": [0], "val": [], "test": []})

    graph = dataset_class(root=str(tmp_path))[0]
    assert float(tlx.convert_to_numpy(graph.y)[0, 0]) == label


@pytest.mark.parametrize("dataset_class,dataset_dir", [
    (Mutagenicity, "mutagenicity"),
    (GraphTokenizerDBLP, "dblp"),
])
@pytest.mark.parametrize("label", [1.5, -1, 2, float("nan"), float("inf")])
def test_binary_classification_labels_fail_fast_when_not_a_valid_class(
        tmp_path, dataset_class, dataset_dir, label):
    _write_raw(tmp_path, dataset_dir, [
        (_graph({"node_token_ids": [[5], [9]]}, {}), label),
    ], {"train": [0], "val": [], "test": []})

    with pytest.raises(ValueError) as error:
        dataset_class(root=str(tmp_path))

    message = str(error.value)
    assert dataset_class.display_name in message
    assert "label" in message


@pytest.mark.parametrize("label", [0.37, -1.42])
def test_aqsol_preserves_finite_float_labels(tmp_path, label):
    _write_raw(tmp_path, "aqsol", [
        (_graph({"feat": [6, 8]}, {"feat": [1]}), label),
    ], {"train": [0], "val": [], "test": []})

    graph = AQSOL(root=str(tmp_path))[0]
    assert float(tlx.convert_to_numpy(graph.y)[0, 0]) == pytest.approx(label)


@pytest.mark.parametrize("label", [float("nan"), float("inf"), float("-inf")])
def test_aqsol_rejects_non_finite_labels(tmp_path, label):
    _write_raw(tmp_path, "aqsol", [
        (_graph({"feat": [6, 8]}, {"feat": [1]}), label),
    ], {"train": [0], "val": [], "test": []})

    with pytest.raises(ValueError, match="AQSOL.*finite"):
        AQSOL(root=str(tmp_path))


def test_aqsol_converts_feat_tokens_and_solubility_label(tmp_path):
    _write_raw(tmp_path, "aqsol", [
        (_graph({"feat": [6, 8]}, {"feat": [1]}), -2.5),
    ], {"train": [0], "val": [], "test": []})

    dataset = AQSOL(root=str(tmp_path))
    graph = dataset[0]

    assert dataset.get_idx_split() == {"train": [0], "val": [], "test": []}
    assert dataset.num_tasks == 1
    assert tlx.convert_to_numpy(graph.x).tolist() == [13, 17]
    assert tlx.convert_to_numpy(graph.edge_attr).tolist() == [2]
    assert float(tlx.convert_to_numpy(graph.y)[0, 0]) == -2.5


def test_aqsol_fails_fast_without_real_bond_features(tmp_path):
    _write_raw(tmp_path, "aqsol", [
        (_graph({"feat": [6, 8]}, {}), -2.5),
    ], {"train": [0], "val": [], "test": []})

    with pytest.raises(ValueError, match="edge features"):
        AQSOL(root=str(tmp_path))


def test_dataset_rejects_overlapping_official_splits(tmp_path):
    samples = [
        (_graph({"node_token_ids": [[5], [9]]}, {"edge_token_ids": [[2]]}), 0),
        (_graph({"node_token_ids": [[6], [8]]}, {"edge_token_ids": [[3]]}), 1),
    ]
    _write_raw(tmp_path, "mutagenicity", samples,
               {"train": [0], "val": [0], "test": [1]})

    with pytest.raises(ValueError, match="overlaps"):
        Mutagenicity(root=str(tmp_path))


@pytest.mark.parametrize("dataset_class,dataset_dir,sample", [
    (Mutagenicity, "mutagenicity", (_graph({"node_token_ids": [[5], [9]]}, {}), 1)),
    (GraphTokenizerDBLP, "dblp", (_graph({"node_token_ids": [[7], [11]]}, {}), 0)),
    (AQSOL, "aqsol", (_graph({"feat": [6, 8]}, {"feat": [1]}), -2.5)),
])
def test_verified_bundle_materializes_real_release_directory_names(
        tmp_path, monkeypatch, dataset_class, dataset_dir, sample):
    module = _download_module()
    bundle = tmp_path / "bundle.tar.gz"
    _write_bundle(bundle, {dataset_dir: [sample]})
    monkeypatch.setattr(module, "PAPER_DATA_BUNDLE_SHA256", module.sha256_file(bundle))
    monkeypatch.setenv(module.DATA_BUNDLE_ENV, str(bundle))
    raw_dir = tmp_path / "data" / dataset_class.name / "raw"

    module.materialize_paper_dataset(
        dataset_class.name, dataset_class.aliases, raw_dir, tmp_path / "data",
        allow_download=False)

    assert (raw_dir / "data.pkl").is_file()
    assert (raw_dir / "train_index.json").read_text(encoding="utf-8") == "[0]"


@pytest.mark.parametrize("dataset_class,dataset_dir,sample", [
    (Mutagenicity, "mutagenicity", (_graph({"node_token_ids": [[5], [9]]}, {}), 1)),
    (GraphTokenizerDBLP, "dblp", (_graph({"node_token_ids": [[7], [11]]}, {}), 0)),
    (AQSOL, "aqsol", (_graph({"feat": [6, 8]}, {"feat": [1]}), -2.5)),
])
def test_dataset_initialization_reuses_prepared_bundle_without_network(
        tmp_path, monkeypatch, dataset_class, dataset_dir, sample):
    bundle = tmp_path / "bundle.tar.gz"
    _write_bundle(bundle, {dataset_dir: [sample]})
    monkeypatch.setattr(
        graph_tokenizer_download,
        "PAPER_DATA_BUNDLE_SHA256",
        graph_tokenizer_download.sha256_file(bundle),
    )
    monkeypatch.setenv(graph_tokenizer_download.DATA_BUNDLE_ENV, str(bundle))
    monkeypatch.setattr(
        "gammagl.data.download.download_google_url",
        lambda *_args, **_kwargs: pytest.fail("prepared bundle must not download"),
    )
    root = tmp_path / "data"
    graph_tokenizer_download.materialize_paper_dataset(
        dataset_class.name, dataset_class.aliases,
        root / dataset_class.name / "raw", root, allow_download=False)

    dataset = dataset_class(root=str(root))

    assert len(dataset) == 1


def test_missing_or_corrupted_bundle_fails_without_network(tmp_path, monkeypatch):
    module = _download_module()
    monkeypatch.delenv(module.DATA_BUNDLE_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="prepared"):
        module.materialize_paper_dataset("aqsol", (), tmp_path / "raw", tmp_path)

    archive = tmp_path / "corrupted.tar.gz"
    archive.write_bytes(b"corrupted")
    monkeypatch.setattr(module, "PAPER_DATA_BUNDLE_SHA256", hashlib.sha256(b"expected").hexdigest())
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        module._verify_official_bundle(archive, remove_on_failure=False)


def test_real_release_bundle_smoke_is_opt_in_and_never_downloads(tmp_path):
    """Validate the actual release bundle only when a local bundle is configured."""
    bundle = os.environ.get(graph_tokenizer_download.DATA_BUNDLE_ENV)
    if not bundle:
        pytest.skip(
            "real release bundle not configured "
            f"(set {graph_tokenizer_download.DATA_BUNDLE_ENV})")
    bundle_path = Path(bundle).expanduser()
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Configured real release bundle does not exist: {bundle_path}")

    root = tmp_path / "real_release"
    expectations = (
        (Mutagenicity, "mutagenicity", 2, "binary_classification"),
        (GraphTokenizerDBLP, "dblp", 2, "binary_classification"),
        (AQSOL, "aqsol", 0, "regression"),
    )
    serializer = FrequencyGuidedEulerianSerializer()

    for dataset_class, name, num_classes, task_type in expectations:
        raw_dir = root / name / "raw"
        graph_tokenizer_download.materialize_paper_dataset(
            dataset_class.name, dataset_class.aliases, raw_dir, root,
            allow_download=False)
        dataset = dataset_class(root=str(root))
        splits = dataset.get_idx_split()

        assert set(splits) == {"train", "val", "test"}
        assert all(splits[split] for split in splits)
        graph = dataset[splits["train"][0]]
        node_tokens = tlx.convert_to_numpy(graph.x)
        edge_tokens = tlx.convert_to_numpy(graph.edge_attr)
        labels = tlx.convert_to_numpy(graph.y)

        assert node_tokens.ndim == edge_tokens.ndim == 1
        assert node_tokens.dtype.kind in "iu"
        assert edge_tokens.dtype.kind in "iu"
        assert node_tokens.size == graph.num_nodes
        assert edge_tokens.size == tlx.convert_to_numpy(graph.edge_index).shape[1]
        assert node_tokens.size and edge_tokens.size
        assert node_tokens.min() >= 0 and edge_tokens.min() >= 0
        assert serializer.serialize(graph).token_ids

        assert dataset.task_type == task_type
        assert dataset.num_classes == num_classes
        assert labels.shape == (1, 1)
        assert labels.dtype.kind == "f"
        if num_classes:
            assert float(labels[0, 0]).is_integer()
            assert 0 <= int(labels[0, 0]) < num_classes
        else:
            assert dataset.num_tasks == 1
