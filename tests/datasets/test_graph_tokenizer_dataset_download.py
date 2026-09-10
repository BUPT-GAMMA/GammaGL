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
def test_materialize_cached_dataset_bundle(
        tmp_path, monkeypatch, dataset_name, data_filename):
    module = _load_download_module()
    bundle_dir = tmp_path / "release" / "data" / dataset_name
    bundle_dir.mkdir(parents=True)
    (bundle_dir / data_filename).write_bytes(pickle.dumps([dataset_name]))
    for split in ("train", "val", "test"):
        (bundle_dir / f"{split}_index.json").write_text("[]", encoding="utf-8")
    monkeypatch.setenv(module.DATA_BUNDLE_ENV, str(tmp_path / "release"))

    raw_dir = module.materialize_paper_dataset(
        dataset_name=dataset_name,
        aliases=(),
        raw_dir=tmp_path / "datasets" / dataset_name / "raw",
        cache_root=tmp_path / "datasets",
        allow_download=True,
    )

    assert raw_dir.joinpath(data_filename).is_file()
    assert json.loads(raw_dir.joinpath("train_index.json").read_text()) == []
