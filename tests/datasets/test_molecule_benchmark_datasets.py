import gzip
import json
import pickle

import pytest
import tensorlayerx as tlx

from gammagl.datasets import OGBGMolHIV, PeptidesStruct, QM9


def _write_raw_dataset(root, name, samples, compressed=False):
    raw_dir = root / name / "raw"
    raw_dir.mkdir(parents=True)
    opener = gzip.open if compressed else open
    with opener(raw_dir / ("data.pkl.gz" if compressed else "data.pkl"), "wb") as file:
        pickle.dump(samples, file)
    for split, indices in {"train": [0], "val": [], "test": []}.items():
        (raw_dir / f"{split}_index.json").write_text(json.dumps(indices))


def test_qm9_loads_a_minimal_local_sample(tmp_path):
    _write_raw_dataset(tmp_path, "qm9", [{
        "edge_index": [[0], [1]], "x": [6, 8], "edge_attr": [1],
        "properties": {name: 0.0 for name in QM9.label_keys},
    }])

    dataset = QM9(root=str(tmp_path))

    assert len(dataset) == 1
    assert tlx.get_tensor_shape(dataset[0].y) == [1, 16]


def test_ogbg_molhiv_loads_a_minimal_local_sample(tmp_path):
    _write_raw_dataset(tmp_path, "ogbg-molhiv", [(
        {"edges": [[0], [1]], "node_type_ids": [6, 8], "edge_type_ids": [1]},
        [1],
    )])

    dataset = OGBGMolHIV(root=str(tmp_path))

    assert len(dataset) == 1
    assert tlx.get_tensor_shape(dataset[0].y) == [1, 1]


def test_peptides_struct_loads_a_minimal_local_sample(tmp_path):
    _write_raw_dataset(tmp_path, "peptides-struct", [(
        {"edges": [[0], [1]], "node_token_ids": [5, 9], "edge_token_ids": [2]},
        [0.0] * 11,
    )], compressed=True)

    dataset = PeptidesStruct(root=str(tmp_path))

    assert len(dataset) == 1
    assert tlx.get_tensor_shape(dataset[0].y) == [1, 11]
