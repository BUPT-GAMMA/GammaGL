import gzip
import json
import pickle

import pytest
import tensorlayerx as tlx

from gammagl.datasets import OGBGMolHIV, PeptidesStruct, QM9


def _write_preprocessed_raw(root, folder_name, samples, splits, compressed=False):
    raw_dir = root / folder_name / "raw"
    raw_dir.mkdir(parents=True)
    data_name = "data.pkl.gz" if compressed else "data.pkl"
    opener = gzip.open if compressed else open
    with opener(raw_dir / data_name, "wb") as f:
        pickle.dump(samples, f)
    for split_name, indices in splits.items():
        (raw_dir / f"{split_name}_index.json").write_text(json.dumps(indices), encoding="utf-8")


def test_qm9_loads_preprocessed_raw_files(tmp_path):
    properties = {
        "mu": 0.1,
        "alpha": 0.2,
        "homo": 0.3,
        "lumo": 0.4,
        "gap": 0.5,
        "r2": 0.6,
        "zpve": 0.7,
        "u0": 0.8,
        "u298": 0.9,
        "h298": 1.0,
        "g298": 1.1,
        "cv": 1.2,
        "u0_atom": 1.3,
        "u298_atom": 1.4,
        "h298_atom": 1.5,
        "g298_atom": 1.6,
    }
    samples = [
        {
            "edge_index": [[0], [1]],
            "x": [6, 8],
            "edge_attr": [1],
            "properties": properties,
        },
        {
            "edge_index": [[0], [1]],
            "x": [7, 6],
            "edge_attr": [2],
            "properties": {key: value + 1.0 for key, value in properties.items()},
        },
    ]
    _write_preprocessed_raw(tmp_path, "qm9", samples, {"train": [0], "val": [1], "test": []})

    dataset = QM9(root=str(tmp_path))

    assert len(dataset) == 2
    assert dataset.num_tasks == 16
    assert dataset.metric == "mae"
    assert dataset.get_idx_split() == {"train": [0], "val": [1], "test": []}
    assert len(dataset.get_split("train")) == 1
    assert tlx.convert_to_numpy(dataset[0].edge_index).tolist() == [[0], [1]]
    assert tlx.convert_to_numpy(dataset[0].y).tolist()[0][:3] == pytest.approx(
        [0.1, 0.2, 0.3])
    assert tlx.convert_to_numpy(dataset[0].x).tolist() == [13, 17]
    assert tlx.convert_to_numpy(dataset[0].edge_attr).tolist() == [2]


def test_qm9_converts_official_attr_columns_to_paper_token_ids(tmp_path):
    properties = {
        name: float(index) for index, name in enumerate(QM9.label_keys)
    }
    node_attr = [
        [0, 0, 0, 0, 0, 6],
        [0, 0, 0, 0, 0, 8],
    ]
    edge_attr = [[0, 0, 1, 0]]
    samples = [{
        "edge_index": [[0], [1]],
        "attr": node_attr,
        "edge_attr": edge_attr,
        "properties": properties,
    }]
    _write_preprocessed_raw(
        tmp_path, "qm9", samples,
        {"train": [0], "val": [], "test": []},
    )

    dataset = QM9(root=str(tmp_path))

    assert tlx.convert_to_numpy(dataset[0].x).tolist() == [13, 17]
    assert tlx.convert_to_numpy(dataset[0].edge_attr).tolist() == [6]


def test_qm9_downloads_released_data_through_dataset_api(tmp_path, monkeypatch):
    properties = {
        name: float(index)
        for index, name in enumerate(QM9.label_keys)
    }
    bundle_dir = tmp_path / "paper-release" / "data" / "qm9"
    bundle_dir.mkdir(parents=True)
    with (bundle_dir / "data.pkl").open("wb") as handle:
        pickle.dump([{
            "edge_index": [[0], [1]],
            "x": [6, 8],
            "edge_attr": [1],
            "properties": properties,
        }], handle)
    for split, indices in {"train": [0], "val": [], "test": []}.items():
        (bundle_dir / f"{split}_index.json").write_text(
            json.dumps(indices), encoding="utf-8")
    monkeypatch.setenv(
        "GAMMAGL_GRAPH_TOKENIZER_DATA_BUNDLE",
        str(tmp_path / "paper-release"),
    )

    dataset = QM9(root=str(tmp_path / "datasets"))

    assert len(dataset) == 1
    assert dataset.get_idx_split() == {"train": [0], "val": [], "test": []}


def test_qm9_rejects_missing_paper_target_property(tmp_path):
    samples = [
        {
            "edge_index": [[0], [1]],
            "x": [6, 8],
            "edge_attr": [1],
            "properties": {"mu": 0.1},
        }
    ]
    _write_preprocessed_raw(tmp_path, "qm9", samples, {"train": [0], "val": [], "test": []})

    with pytest.raises(ValueError, match="16 QM9 properties"):
        QM9(root=str(tmp_path))


def test_dataset_rejects_overlapping_split_indices(tmp_path):
    samples = [
        ({"edges": [[0], [1]], "node_type_ids": [6, 8], "edge_type_ids": [1]}, [1]),
        ({"edges": [[0], [1]], "node_type_ids": [6, 6], "edge_type_ids": [1]}, [0]),
    ]
    _write_preprocessed_raw(tmp_path, "ogbg-molhiv", samples, {"train": [0], "val": [0], "test": [1]})

    with pytest.raises(ValueError, match="overlaps"):
        OGBGMolHIV(root=str(tmp_path))


def test_ogbg_molhiv_loads_alias_directory_and_label(tmp_path):
    samples = [
        ({"edges": [[0, 1], [1, 0]], "node_type_ids": [6, 6], "edge_type_ids": [1, 1]}, [0]),
        ({"edges": [[0], [1]], "node_type_ids": [6, 8], "edge_type_ids": [2]}, [1]),
    ]
    _write_preprocessed_raw(tmp_path, "ogbg-molhiv", samples, {"train": [0], "val": [], "test": [1]})

    dataset = OGBGMolHIV(root=str(tmp_path))

    assert len(dataset) == 2
    assert dataset.name == "ogbg-molhiv"
    assert dataset.num_tasks == 1
    assert dataset.metric == "rocauc"
    assert dataset.get_idx_split()["test"] == [1]
    assert tlx.convert_to_numpy(dataset[1].y).tolist() == [[1.0]]


def test_peptides_struct_loads_compressed_raw_files(tmp_path):
    samples = [
        (
            {
                "edges": [[0, 1], [1, 0]],
                "node_token_ids": [[5], [9]],
                "edge_token_ids": [[2], [2]],
            },
            [float(i) for i in range(11)],
        )
    ]
    _write_preprocessed_raw(tmp_path, "peptides-struct", samples, {"train": [0], "val": [], "test": []}, compressed=True)

    dataset = PeptidesStruct(root=str(tmp_path))

    assert len(dataset) == 1
    assert dataset.num_tasks == 11
    assert dataset.metric == "average_mae"
    assert tlx.convert_to_numpy(dataset[0].x).tolist() == [5, 9]
    assert tlx.convert_to_numpy(dataset[0].y).shape == (1, 11)
