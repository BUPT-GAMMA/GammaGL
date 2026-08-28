import importlib.util
import json
import pickle
from pathlib import Path

import pytest


def _load_download_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "gammagl"
        / "datasets"
        / "_graph_tokenizer_download.py"
    )
    spec = importlib.util.spec_from_file_location(
        "graph_tokenizer_download_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("dataset_name", "data_filename"),
    (
        ("qm9", "data.pkl"),
        ("ogbg-molhiv", "data.pkl"),
        ("peptides-struct", "data.pkl.gz"),
    ),
)
def test_materialize_dataset_from_cached_official_bundle(
        tmp_path, monkeypatch, dataset_name, data_filename):
    module = _load_download_module()
    bundle_root = tmp_path / "release"
    source_dir = bundle_root / "data" / dataset_name
    source_dir.mkdir(parents=True)
    (source_dir / data_filename).write_bytes(pickle.dumps([dataset_name]))
    for split, indices in {"train": [0], "val": [], "test": []}.items():
        (source_dir / f"{split}_index.json").write_text(
            json.dumps(indices), encoding="utf-8")

    monkeypatch.setenv(module.DATA_BUNDLE_ENV, str(bundle_root))
    raw_dir = tmp_path / "datasets" / dataset_name / "raw"

    copied = module.materialize_paper_dataset(
        dataset_name=dataset_name,
        aliases=(dataset_name.replace("-", "_"),),
        raw_dir=raw_dir,
        cache_root=tmp_path / "datasets",
    )

    assert copied == raw_dir
    assert (raw_dir / data_filename).read_bytes() == (source_dir / data_filename).read_bytes()
    assert json.loads((raw_dir / "train_index.json").read_text(encoding="utf-8")) == [0]


def test_materialize_dataset_rejects_bundle_without_official_splits(tmp_path, monkeypatch):
    module = _load_download_module()
    bundle_root = tmp_path / "release"
    source_dir = bundle_root / "data" / "qm9"
    source_dir.mkdir(parents=True)
    (source_dir / "data.pkl").write_bytes(pickle.dumps([]))
    monkeypatch.setenv(module.DATA_BUNDLE_ENV, str(bundle_root))

    with pytest.raises(FileNotFoundError, match="official split"):
        module.materialize_paper_dataset(
            dataset_name="qm9",
            aliases=(),
            raw_dir=tmp_path / "qm9" / "raw",
            cache_root=tmp_path,
        )
