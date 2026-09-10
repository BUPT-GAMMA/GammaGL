import hashlib
import os
import shutil
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Tuple


DATA_BUNDLE_ENV = 'GAMMAGL_GRAPH_TOKENIZER_DATA_BUNDLE'
PAPER_DATA_BUNDLE_ID = '10etZF9OnV569_Fp7tpdMUVEH9eZECKdW'
PAPER_DATA_BUNDLE_FILENAME = 'GraphToenizerDataset.tar.gz'
# SHA-256 of the bytes published by the fixed Google Drive release above.
PAPER_DATA_BUNDLE_SHA256 = (
    '5c437c3c0d4b7278379c0e70d57f98148e5c815d753d8cf68e2a45952bcce459')
_REQUIRED_SPLITS = ('train_index.json', 'val_index.json', 'test_index.json')
_OPTIONAL_FILES = ('token_mappings.json',)


def materialize_paper_dataset(
        dataset_name: str,
        aliases: Iterable[str],
        raw_dir,
        cache_root,
        allow_download: bool = False,
) -> Path:
    """Copy one dataset from a verified, already-prepared paper bundle."""
    raw_dir = Path(raw_dir)
    cache_root = Path(cache_root)
    source = _resolve_bundle_source(cache_root, allow_download=allow_download)
    bundle_root = source if source.is_dir() else _extract_archive(source, cache_root)
    dataset_dir = _find_dataset_dir(bundle_root, dataset_name, tuple(aliases))

    data_file = _find_data_file(dataset_dir)
    missing_splits = [name for name in _REQUIRED_SPLITS if not (dataset_dir / name).is_file()]
    if data_file is None or missing_splits:
        details = []
        if data_file is None:
            details.append('data.pkl or data.pkl.gz')
        if missing_splits:
            details.append(f"official split files: {', '.join(missing_splits)}")
        raise FileNotFoundError(
            f"Released {dataset_name} directory is incomplete; missing {'; '.join(details)}.")

    raw_dir.mkdir(parents=True, exist_ok=True)
    for source_file in (data_file, *(dataset_dir / name for name in _REQUIRED_SPLITS)):
        shutil.copy2(source_file, raw_dir / source_file.name)
    for name in _OPTIONAL_FILES:
        source_file = dataset_dir / name
        if source_file.is_file():
            shutil.copy2(source_file, raw_dir / name)
    return raw_dir


@contextmanager
def _bundle_lock(cache_dir: Path):
    """Serialize installation of the shared release archive."""
    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - paper runs are Linux-only.
        raise RuntimeError("GraphTokenizer paper preparation requires fcntl locking.") from error
    lock_path = cache_dir / ".bundle.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _resolve_bundle_source(cache_root: Path, allow_download: bool = False) -> Path:
    configured = os.environ.get(DATA_BUNDLE_ENV)
    if configured:
        if not allow_download:
            raise FileNotFoundError(
                "GraphTokenizer paper data must be prepared by the single "
                "data-preparation process before starting training workers.")
        source = Path(configured).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(
                f"{DATA_BUNDLE_ENV} points to a missing path: {source}")
        if source.is_file():
            _verify_official_bundle(source, remove_on_failure=False)
        return source

    cache_dir = cache_root / '.graph_tokenizer_release'
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / PAPER_DATA_BUNDLE_FILENAME
    with _bundle_lock(cache_dir):
        if archive.exists():
            _verify_official_bundle(archive, remove_on_failure=False)
        elif not allow_download:
            raise FileNotFoundError(
                "GraphTokenizer paper data must be prepared by the single "
                "data-preparation process before starting training workers.")
        else:
            from gammagl.data.download import download_google_url

            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{archive.name}.{os.getpid()}.", suffix=".tmp", dir=cache_dir)
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                download_google_url(
                    PAPER_DATA_BUNDLE_ID,
                    str(cache_dir),
                    temporary.name,
                )
                _verify_official_bundle(temporary, remove_on_failure=False)
                os.replace(temporary, archive)
            except Exception:
                if temporary.exists():
                    temporary.unlink()
                raise
    return archive


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the streaming SHA-256 digest of a file's actual bytes."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_official_bundle(archive: Path, remove_on_failure: bool) -> str:
    actual = sha256_file(archive)
    if actual == PAPER_DATA_BUNDLE_SHA256:
        return actual
    if remove_on_failure and archive.exists():
        archive.unlink()
    raise RuntimeError(
        'GraphTokenizer paper bundle SHA-256 mismatch: expected '
        f'{PAPER_DATA_BUNDLE_SHA256}, got {actual}.')


def _extract_archive(archive: Path, cache_root: Path) -> Path:
    digest = _verify_official_bundle(archive, remove_on_failure=False)[:16]
    destination = cache_root / '.graph_tokenizer_release' / f'extracted-{digest}'
    marker = destination / '.complete'
    if marker.is_file():
        return destination

    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        _extract_zip(archive, destination)
    elif tarfile.is_tarfile(archive):
        _extract_tar(archive, destination)
    else:
        raise ValueError(
            f'Unsupported paper data bundle format: {archive}. Expected ZIP or TAR.')
    marker.touch()
    return destination


def _safe_member_path(destination: Path, member_name: str) -> Path:
    member_path = (destination / member_name).resolve()
    try:
        member_path.relative_to(destination.resolve())
    except ValueError as error:
        raise ValueError(f'Unsafe path in paper data bundle: {member_name}') from error
    return member_path


def _extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = _safe_member_path(destination, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(member) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)


def _extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f'Links are not allowed in paper data bundle: {member.name}')
            target = _safe_member_path(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f'Cannot extract paper data bundle member: {member.name}')
            with source, target.open('wb') as output:
                shutil.copyfileobj(source, output)


def _find_dataset_dir(
        bundle_root: Path,
        dataset_name: str,
        aliases: Tuple[str, ...],
) -> Path:
    accepted_names = {_normalize_name(dataset_name)}
    accepted_names.update(_normalize_name(alias) for alias in aliases)
    candidates = []
    for data_file in (*bundle_root.rglob('data.pkl'), *bundle_root.rglob('data.pkl.gz')):
        parent = data_file.parent
        if _normalize_name(parent.name) in accepted_names:
            candidates.append(parent)
    unique_candidates = sorted(set(candidates), key=lambda path: (len(path.parts), str(path)))
    if not unique_candidates:
        names = ', '.join(sorted(accepted_names))
        raise FileNotFoundError(
            f"Cannot find {dataset_name} ({names}) in released paper data bundle {bundle_root}.")
    return unique_candidates[0]


def _find_data_file(dataset_dir: Path):
    for name in ('data.pkl', 'data.pkl.gz'):
        path = dataset_dir / name
        if path.is_file():
            return path
    return None


def _normalize_name(value: str) -> str:
    return str(value).strip().lower().replace('_', '-')
