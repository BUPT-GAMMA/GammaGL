import importlib.util
import hashlib
import json
import pickle
import threading
import time
import zipfile
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
        allow_download=True,
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
            allow_download=True,
        )


def test_download_bundle_uses_actual_file_sha256(tmp_path, monkeypatch):
    module = _load_download_module()
    archive = tmp_path / "official.tar.gz"
    archive.write_bytes(b"official release bytes")
    expected = hashlib.sha256(b"official release bytes").hexdigest()
    monkeypatch.setattr(module, "PAPER_DATA_BUNDLE_SHA256", expected)

    assert module.sha256_file(archive) == expected
    assert module._verify_official_bundle(archive, remove_on_failure=True) == expected


def test_cached_bundle_checksum_mismatch_is_deleted_and_hard_fails(tmp_path, monkeypatch):
    module = _load_download_module()
    archive = (tmp_path / ".graph_tokenizer_release"
               / module.PAPER_DATA_BUNDLE_FILENAME)
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"tampered release bytes")
    monkeypatch.setattr(module, "PAPER_DATA_BUNDLE_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        module._resolve_bundle_source(tmp_path)
    assert archive.exists()


def test_training_materialization_requires_prepared_bundle(tmp_path, monkeypatch):
    module = _load_download_module()
    monkeypatch.delenv(module.DATA_BUNDLE_ENV, raising=False)

    with pytest.raises(FileNotFoundError, match="prepared by the single data-preparation"):
        module.materialize_paper_dataset(
            dataset_name="qm9",
            aliases=(),
            raw_dir=tmp_path / "qm9" / "raw",
            cache_root=tmp_path,
        )


def test_preparation_installs_verified_bundle_atomically(tmp_path, monkeypatch):
    module = _load_download_module()
    payload = b"official release bytes"
    monkeypatch.setattr(module, "PAPER_DATA_BUNDLE_SHA256", hashlib.sha256(payload).hexdigest())

    def fake_download(_file_id, destination, filename):
        path = Path(destination) / filename
        path.write_bytes(payload)
        return str(path)

    import gammagl.data.download as download
    monkeypatch.setattr(download, "download_google_url", fake_download)

    archive = module._resolve_bundle_source(tmp_path, allow_download=True)

    assert archive.name == module.PAPER_DATA_BUNDLE_FILENAME
    assert archive.read_bytes() == payload
    assert not list(archive.parent.glob(f".{archive.name}.*.tmp"))


def test_concurrent_preparation_has_one_final_bundle_writer(tmp_path, monkeypatch):
    module = _load_download_module()
    payload = b"official release bytes"
    monkeypatch.setattr(module, "PAPER_DATA_BUNDLE_SHA256", hashlib.sha256(payload).hexdigest())
    calls = []
    calls_lock = threading.Lock()

    def fake_download(_file_id, destination, filename):
        with calls_lock:
            calls.append(filename)
        time.sleep(0.05)
        path = Path(destination) / filename
        path.write_bytes(payload)
        return str(path)

    import gammagl.data.download as download
    monkeypatch.setattr(download, "download_google_url", fake_download)
    results = []
    failures = []

    def prepare():
        try:
            results.append(module._resolve_bundle_source(tmp_path, allow_download=True))
        except Exception as error:  # pragma: no cover - asserted below
            failures.append(error)

    workers = [threading.Thread(target=prepare) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert not failures
    assert len(calls) == 1
    assert len(results) == 2
    assert results[0] == results[1]
    assert results[0].read_bytes() == payload


def test_failed_downloader_keeps_existing_verified_shared_bundle(tmp_path, monkeypatch):
    module = _load_download_module()
    payload = b"official release bytes"
    archive = tmp_path / ".graph_tokenizer_release" / module.PAPER_DATA_BUNDLE_FILENAME
    archive.parent.mkdir(parents=True)
    archive.write_bytes(payload)
    monkeypatch.setattr(module, "PAPER_DATA_BUNDLE_SHA256", hashlib.sha256(payload).hexdigest())

    import gammagl.data.download as download
    monkeypatch.setattr(
        download, "download_google_url",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("network failed")),
    )

    assert module._resolve_bundle_source(tmp_path, allow_download=True) == archive
    assert archive.read_bytes() == payload


def test_explicit_archive_checksum_mismatch_hard_fails(tmp_path, monkeypatch):
    module = _load_download_module()
    archive = tmp_path / "untrusted.tar.gz"
    archive.write_bytes(b"not the official release")
    monkeypatch.setenv(module.DATA_BUNDLE_ENV, str(archive))
    monkeypatch.setattr(module, "PAPER_DATA_BUNDLE_SHA256", "f" * 64)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        module._resolve_bundle_source(tmp_path, allow_download=True)
    assert archive.exists()


def test_bundle_extraction_rejects_path_traversal(tmp_path):
    module = _load_download_module()
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside", b"unsafe")

    with pytest.raises(ValueError, match="Unsafe path"):
        module._extract_zip(archive, tmp_path / "extracted")
    assert not (tmp_path / "outside").exists()
