"""Portable, versioned artifacts for the native GraphTokenizer trainer.

Tokenizer pickles and torch checkpoints must come from trusted runs. Hashes
detect accidental mixing/corruption; they do not authenticate untrusted files.
"""
from datetime import datetime, timezone
import errno
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

ARTIFACT_VERSION = 1


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def model_configuration(model):
    """Save actual constructor values, including settings absent from CLI presets."""
    from gammagl.models.graph_bert import GraphBERT
    from gammagl.models.graph_gte import GraphGTE
    if not isinstance(model, (GraphBERT, GraphGTE)):
        raise ValueError('Portable run artifacts require GraphBERT or GraphGTE.')
    fields = ['vocab_size', 'output_dim', 'hidden_size', 'num_hidden_layers',
              'num_attention_heads', 'intermediate_size', 'max_position_embeddings',
              'dropout_rate', 'attention_dropout_rate', 'layer_norm_eps',
              'pad_token_id', 'type_vocab_size', 'task_type', 'pooling']
    encoder = 'gte' if isinstance(model, GraphGTE) else 'bert'
    if encoder == 'gte':
        fields += ['rope_theta', 'rope_scaling_factor']
    kwargs = {name: getattr(model, name) for name in fields}
    kwargs['task_dropout'] = model.task_dropout_rate
    config = dict(encoder=encoder, kwargs=kwargs)
    if encoder == 'bert':
        config['model_protocol_version'] = model.model_protocol_version
    return config


def code_provenance(source):
    paths = [*Path(__file__).parent.glob('*.py'),
             *source.glob('gammagl/models/*graph*.py'),
             *source.glob('gammagl/transforms/graph*.py'),
             *source.glob('gammagl/datasets/*graph_tokenizer*.py'),
             *(source / 'gammagl/datasets' / f'{name}.py'
               for name in ('aqsol', 'mutagenicity'))]
    files = {str(path.relative_to(source)): digest(path) for path in sorted(set(paths))}
    commit, dirty = None, None
    try:
        def git(*args):
            return subprocess.check_output(['git', '-C', str(source), *args],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        if Path(git('rev-parse', '--show-toplevel')).resolve() == source.resolve():
            commit = git('rev-parse', 'HEAD')
            dirty = bool(git('status', '--porcelain'))
    except (OSError, subprocess.CalledProcessError):
        pass
    return dict(commit=commit, dirty=dirty, files_sha256=files,
                version_source='git_and_source_hashes' if commit else 'source_hashes')


def environment(backend):
    packages = {}
    for name in ('tensorlayerx', 'numpy', 'torch', 'tensorflow-cpu', 'tensorflow',
                 'dgl', 'huggingface-hub', 'safetensors'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    distribution = importlib.metadata.distribution('tensorlayerx')
    direct_url = json.loads(distribution.read_text('direct_url.json') or '{}')
    return dict(backend=backend, python=sys.version, packages=packages,
                tensorlayerx_runtime_version=getattr(sys.modules.get('tensorlayerx'), '__version__', None),
                tensorlayerx_vcs=direct_url.get('vcs_info'),
                backend_runtime_version=getattr(sys.modules.get(backend), '__version__', None),
                cuda_runtime=getattr(getattr(sys.modules.get('torch'), 'version', None), 'cuda', None))


def data_manifest(args, spec, encoded, synthetic=None):
    directory = Path(args.data_root) / spec.storage_name
    files = {} if args.smoke else {
        str(path.relative_to(directory)): digest(path)
        for subdir in ('raw', 'processed') for path in sorted((directory / subdir).glob('*'))
        if path.is_file() and (subdir == 'raw' or '_v2.' in path.name)}
    result = dict(dataset=spec.name, source_kind='synthetic' if args.smoke else 'release_files',
                  files_sha256=files,
                  split_graph_counts={s: len({r['graph_id'] for r in rows}) for s, rows in encoded.items()},
                  encoded_record_counts={s: len(rows) for s, rows in encoded.items()})
    if synthetic is not None:
        result['synthetic_graphs'] = {s: [vars(graph) for graph in graphs] for s, graphs in synthetic.items()}
    return result


def _read_bundle(directory, expected):
    descriptor = json.loads((directory / 'bundle.json').read_text())
    actual = hashlib.sha256(json_bytes(descriptor)).hexdigest()
    if actual != expected:
        raise ValueError('Preprocessing bundle manifest hash mismatch.')
    if descriptor.get('artifact_version') != ARTIFACT_VERSION:
        raise ValueError('Unsupported run artifact version.')
    if set(descriptor['files_sha256']) != {'tokenizer.pkl', 'normalizer.json', 'data_manifest.json'}:
        raise ValueError('Incomplete preprocessing bundle file manifest.')
    for name, checksum in descriptor['files_sha256'].items():
        if digest(directory / name) != checksum:
            raise ValueError(f'Preprocessing file hash mismatch: {name}')
    return descriptor


def save_run(directory, args, spec, model, tokenizer, normalizer, encoded, runtime, *, synthetic=None):
    """Write immutable paired inputs and a per-seed manifest before training."""
    from examples.graph_tokenizer import graph_tokenizer_trainer as trainer
    trainer.require_finite_metrics(normalizer, 'artifact normalizer')
    config = model_configuration(model)
    files = {'tokenizer.pkl': pickle.dumps(tokenizer, protocol=pickle.HIGHEST_PROTOCOL),
             'normalizer.json': json_bytes(normalizer),
             'data_manifest.json': json_bytes(data_manifest(args, spec, encoded, synthetic))}
    descriptor = dict(artifact_version=ARTIFACT_VERSION, model_config=config,
                      tokenizer_fingerprint=tokenizer.fingerprint(), fit_scope=tokenizer.fit_scope,
                      files_sha256={name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    fingerprint = hashlib.sha256(json_bytes(descriptor)).hexdigest()
    bundle = directory.parent / 'preprocessing' / fingerprint
    if bundle.exists():
        _read_bundle(bundle, fingerprint)
    else:
        bundle.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f'.{fingerprint}.', dir=bundle.parent) as staging:
            staging = Path(staging)
            for name, content in files.items():
                atomic_write(staging / name, content)
            atomic_write(staging / 'bundle.json', json_bytes(descriptor))
            _read_bundle(staging, fingerprint)
            try:
                os.rename(staging, bundle)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                # Another writer may have published the same complete bundle.
                _read_bundle(bundle, fingerprint)
    manifest = dict(artifact_version=ARTIFACT_VERSION,
                    training_protocol=trainer.training_protocol(),
                    model_protocol_version=getattr(model, 'model_protocol_version', None),
                    started_at=datetime.now(timezone.utc).isoformat(),
                    effective_config={k: v for k, v in vars(args).items() if not k.startswith('_')},
                    cli_args=getattr(args, '_cli_args', None), runtime=runtime,
                    environment=environment(runtime['backend']),
                    code=code_provenance(Path(__file__).resolve().parents[2]),
                    model_config=config, model_manifest=model.manifest(),
                    preprocessing=os.path.relpath(bundle, directory),
                    preprocessing_fingerprint=fingerprint)
    atomic_write(directory / 'run_manifest.json', json_bytes(manifest))
    model.artifact_fingerprint = fingerprint
    return manifest


def completed_run(directory, checkpoint, *, pretrain_checkpoint=None):
    """Require every paired file to exist and match before reporting success."""
    manifest = json.loads((directory / 'run_manifest.json').read_text())
    _read_bundle(directory / manifest['preprocessing'], manifest['preprocessing_fingerprint'])
    if (manifest.get('training_protocol') or {}).get('version') == 2 and pretrain_checkpoint is None:
        raise ValueError('Incomplete v2 export: pretraining checkpoint is required.')
    result = dict(manifest='run_manifest.json', manifest_sha256=digest(directory / 'run_manifest.json'),
                checkpoint=checkpoint.name, checkpoint_sha256=digest(checkpoint),
                preprocessing_fingerprint=manifest['preprocessing_fingerprint'])
    if pretrain_checkpoint is not None:
        result.update(pretrain_checkpoint=pretrain_checkpoint.name,
                      pretrain_checkpoint_sha256=digest(pretrain_checkpoint))
    return result


def load_run(directory, device='cpu'):
    """Restore a trusted exported run without raw data, caches or HF downloads.

    Parameters
    ----------
    directory : str or path-like
        Per-seed directory containing summary.json and run_manifest.json. Its
        relative preprocessing bundle must be copied together with the run.
    device : str
        cpu, cuda or cuda:N; TensorFlow device selection must precede any tensor
        operations in the process. The checkpoint backend must match TL_BACKEND.

    Returns
    -------
    dict
        model in evaluation mode, fitted tokenizer, normalizer and run manifest.

    Raises
    ------
    ValueError
        For missing legacy metadata, unsupported versions, altered files,
        incompatible backend, tokenizer, model or label normalization bindings.
    """
    from examples.graph_tokenizer.runtime import configure_device
    # Configure TF before importing TLX/the trainer in a fresh restore process.
    selected = getattr(sys.modules.get('tensorlayerx'), 'BACKEND', None) or os.environ.get('TL_BACKEND') or 'torch'
    runtime = configure_device(selected, device)
    from examples.graph_tokenizer import graph_tokenizer_trainer as trainer
    from gammagl.models.graph_bert import GraphBERT
    from gammagl.models.graph_gte import GraphGTE
    directory = Path(directory).resolve()
    if not (directory / 'run_manifest.json').is_file():
        raise ValueError('Legacy run has no export manifest; its original paired assets are required.')
    summary = json.loads((directory / 'summary.json').read_text())
    trainer.validate_run_result(summary)
    artifacts = summary['artifacts']
    if digest(directory / 'run_manifest.json') != artifacts['manifest_sha256']:
        raise ValueError('Run manifest hash mismatch.')
    manifest = json.loads((directory / 'run_manifest.json').read_text())
    if manifest.get('artifact_version') != ARTIFACT_VERSION:
        raise ValueError('Unsupported run artifact version.')
    if summary.get('training_protocol') != manifest.get('training_protocol'):
        raise ValueError('Run training protocol differs from its manifest.')
    if summary.get('model_protocol_version') != manifest.get('model_protocol_version'):
        raise ValueError('Run model protocol differs from its manifest.')
    if manifest['model_config']['encoder'] == 'bert' and any(
            version != GraphBERT.model_protocol_version for version in (
                manifest.get('model_protocol_version'),
                manifest['model_config'].get('model_protocol_version'),
                manifest['model_manifest'].get('model_protocol_version'))):
        raise ValueError('Unsupported BERT model protocol; use the original code for legacy inference.')
    if ((manifest.get('training_protocol') or {}).get('version') == 2
            and not {'pretrain_checkpoint', 'pretrain_checkpoint_sha256'} <= artifacts.keys()):
        raise ValueError('Incomplete v2 export: pretraining checkpoint metadata is required.')
    if manifest['environment']['backend'] != trainer.tlx.BACKEND:
        raise ValueError('Checkpoint backend differs from the active TLX backend.')
    bundle = (directory / manifest['preprocessing']).resolve()
    if not bundle.is_relative_to(directory.parent):
        raise ValueError('Preprocessing bundle must be inside the experiment directory.')
    descriptor = _read_bundle(bundle, manifest['preprocessing_fingerprint'])
    if descriptor['model_config'] != manifest['model_config']:
        raise ValueError('Model configuration differs from the preprocessing bundle.')
    with (bundle / 'tokenizer.pkl').open('rb') as handle:
        tokenizer = pickle.load(handle)
    if (tokenizer.fingerprint() != descriptor['tokenizer_fingerprint']
            or tokenizer.fit_scope != descriptor['fit_scope']):
        raise ValueError('Tokenizer fingerprint or fit scope mismatch.')
    normalizer = json.loads((bundle / 'normalizer.json').read_text())
    trainer.require_finite_metrics(normalizer, 'restored normalizer')
    config = manifest['model_config']
    model_class = {'bert': GraphBERT, 'gte': GraphGTE}[config['encoder']]
    checkpoint_name = 'best_finetune.npz' if trainer.tlx.BACKEND == 'tensorflow' else 'best_finetune.pt'
    if artifacts['checkpoint'] != checkpoint_name:
        raise ValueError('Unexpected checkpoint filename.')
    checkpoint = directory / checkpoint_name
    if digest(checkpoint) != artifacts['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch.')
    if 'pretrain_checkpoint' in artifacts:
        expected = checkpoint_name.replace('finetune', 'pretrain')
        if artifacts['pretrain_checkpoint'] != expected:
            raise ValueError('Unexpected pretraining checkpoint filename.')
        if digest(directory / expected) != artifacts['pretrain_checkpoint_sha256']:
            raise ValueError('Pretraining checkpoint hash mismatch.')
    # Complete downstream weights include embeddings/MLM/task heads: do not
    # fetch or reapply the encoder's original pretrained initialization.
    model = model_class(**config['kwargs'])
    tokenizer.validate_model_vocab(model.vocab_size)
    model.tokenizer_fingerprint = tokenizer.fingerprint()
    model.tokenizer_fit_scope = tokenizer.fit_scope
    model.artifact_fingerprint = manifest['preprocessing_fingerprint']
    model.training_protocol_version = (manifest.get('training_protocol') or {}).get('version')
    if config['encoder'] == 'gte':
        model.model_id_to_official_id = tokenizer.model_id_to_official_id
    if trainer.tlx.BACKEND == 'torch':
        model = model.to(runtime['resolved_device'])
    trainer.restore_checkpoint(trainer.tlx, checkpoint, model, runtime['resolved_device'])
    trainer.set_model_mode(model, False)
    return dict(model=model, tokenizer=tokenizer, normalizer=normalizer, manifest=manifest)
