import hashlib
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, Tuple


DATA_BUNDLE_ENV = 'GAMMAGL_GRAPH_TOKENIZER_DATA_BUNDLE'
PAPER_DATA_BUNDLE_ID = '10etZF9OnV569_Fp7tpdMUVEH9eZECKdW'
_REQUIRED_SPLITS = ('train_index.json', 'val_index.json', 'test_index.json')
_OPTIONAL_FILES = ('token_mappings.json',)


def materialize_paper_dataset(
        dataset_name: str,
        aliases: Iterable[str],
        raw_dir,
        cache_root,
) -> Path:
    """Download the released paper bundle and copy one dataset into ``raw``."""
    raw_dir = Path(raw_dir)
    cache_root = Path(cache_root)
    source = _resolve_bundle_source(cache_root)
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


def _resolve_bundle_source(cache_root: Path) -> Path:
    configured = os.environ.get(DATA_BUNDLE_ENV)
    if configured:
        source = Path(configured).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(
                f"{DATA_BUNDLE_ENV} points to a missing path: {source}")
        return source

    cache_dir = cache_root / '.graph_tokenizer_release'
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / 'graph_tokenizer_datasets.bundle'
    if not archive.exists():
        from gammagl.data.download import download_google_url

        download_google_url(
            PAPER_DATA_BUNDLE_ID,
            str(cache_dir),
            archive.name,
        )
    return archive


def _extract_archive(archive: Path, cache_root: Path) -> Path:
    stat = archive.stat()
    digest = hashlib.sha256(
        f'{archive.resolve()}:{stat.st_size}:{stat.st_mtime_ns}'.encode('utf-8')
    ).hexdigest()[:16]
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
