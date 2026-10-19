"""Run artifacts must survive relocation and restore without data or HF access."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import tensorlayerx as tlx

from test_graph_tokenizer_tlx_training import trainer


@pytest.mark.parametrize('encoder', ['bert', 'gte'])
@pytest.mark.parametrize('dataset', ['aqsol', 'mutagenicity'])
def test_native_run_exports_relocatable_artifacts_and_fresh_process_predictions(
        trainer, monkeypatch, tmp_path, encoder, dataset):
    original = trainer.make_model
    captured = {}

    def record(*args, **kwargs):
        model = original(*args, **kwargs)
        captured.update(model=model, tokenizer=args[4])
        return model

    monkeypatch.setattr(trainer, 'make_model', record)
    argv = ['--smoke', '--dataset', dataset, '--encoder', encoder, '--device', 'cpu',
            '--seed', '42', '--output-dir', str(tmp_path / 'original')]
    if encoder == 'gte':
        argv.append('--allow-random-gte-init')
    trainer.main(argv)
    group = tmp_path / 'original' / dataset / encoder
    run = group / 'run_01_seed_42'
    manifest_path = run / 'run_manifest.json'
    assert manifest_path.exists(), 'native trainer must export resolved configuration before training'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['effective_config']['seed'] == 42
    assert manifest['effective_config']['max_length'] == 32
    assert manifest['code']['files_sha256']
    assert manifest['environment']['backend'] == tlx.BACKEND
    assert manifest['model_config']['kwargs']['output_dim'] == (2 if dataset == 'mutagenicity' else 1)
    bundle = (run / manifest['preprocessing']).resolve()
    normalizer = json.loads((bundle / 'normalizer.json').read_text())
    assert normalizer == ({'mean': [pytest.approx(0.1)], 'std': [pytest.approx(np.sqrt(0.02 / 3))],
                           'targets': ['solubility']} if dataset == 'aqsol' else None)
    data = json.loads((bundle / 'data_manifest.json').read_text())
    assert data['split_graph_counts'] == {'train': 3, 'val': 1, 'test': 1}
    assert data['source_kind'] == 'synthetic'

    graph = trainer.synthetic_splits(trainer.resolve_dataset(dataset))['test'][0]
    encoded = captured['tokenizer'].encode_graph(graph)
    ids = tlx.convert_to_tensor([encoded.input_ids], dtype=tlx.int64)
    mask = tlx.ones_like(ids)
    expected = tlx.convert_to_numpy(captured['model'](ids, mask, task='supervised'))
    expected_raw = expected if normalizer is None else expected * normalizer['std'] + normalizer['mean']
    moved = tmp_path / 'moved'
    shutil.copytree(group, moved)
    shutil.rmtree(tmp_path / 'original')
    payload = tmp_path / 'request.json'
    payload.write_text(json.dumps({'graph': vars(graph), 'input_ids': encoded.input_ids}))
    script = '''
import json, sys
import numpy as np
from examples.graph_tokenizer.artifacts import load_run
restored = load_run(sys.argv[1], device='cpu')
import tensorlayerx as tlx
request = json.load(open(sys.argv[2]))
encoding = restored['tokenizer'].encode_graph(request['graph'])
assert encoding.input_ids == request['input_ids']
ids = tlx.convert_to_tensor([encoding.input_ids], dtype=tlx.int64)
pred = tlx.convert_to_numpy(restored['model'](ids, tlx.ones_like(ids), task='supervised'))
normalizer = restored['normalizer']
raw = pred if normalizer is None else pred * normalizer['std'] + normalizer['mean']
np.savez(sys.argv[3], pred=pred, raw=raw)
'''
    output = tmp_path / 'predictions.npz'
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1',
           'HF_HOME': str(tmp_path / 'empty-hf-cache'),
           'PYTHONPATH': str(Path(__file__).resolve().parents[2])}
    result = subprocess.run([sys.executable, '-c', script, str(moved / run.name),
                             str(payload), str(output)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    with np.load(output) as actual:
        np.testing.assert_allclose(actual['pred'], expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(actual['raw'], expected_raw, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('filename', ['normalizer.json', 'tokenizer.pkl', 'data_manifest.json'])
def test_restore_rejects_modified_preprocessing_before_constructing_model(trainer, tmp_path, filename):
    from examples.graph_tokenizer.artifacts import load_run
    trainer.main(['--smoke', '--dataset', 'aqsol', '--device', 'cpu', '--seed', '42', '--output-dir', str(tmp_path)])
    run = tmp_path / 'aqsol/bert/run_01_seed_42'
    manifest = json.loads((run / 'run_manifest.json').read_text())
    path = run / manifest['preprocessing'] / filename
    path.write_bytes(path.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='hash|digest|checksum'):
        load_run(run, device='cpu')


@pytest.mark.parametrize('encoder', ['bert', 'gte'])
def test_constructor_export_preserves_nondefault_dropout_norm_and_rope(encoder):
    from examples.graph_tokenizer.artifacts import model_configuration
    from gammagl.models import GraphBERT, GraphGTE
    model_class = GraphBERT if encoder == 'bert' else GraphGTE
    kwargs = dict(vocab_size=12, output_dim=2, hidden_size=4, num_hidden_layers=1,
                  num_attention_heads=1, intermediate_size=8, max_position_embeddings=32,
                  dropout_rate=0.21, attention_dropout_rate=0.31, layer_norm_eps=0.001,
                  pad_token_id=3, type_vocab_size=4, task_type='binary_classification',
                  pooling='cls', task_dropout=0.41)
    if encoder == 'gte':
        kwargs.update(rope_theta=1234.0, rope_scaling_factor=2.5)
    config = model_configuration(model_class(**kwargs))
    assert config == {'encoder': encoder, 'kwargs': kwargs}
    rebuilt = model_class(**config['kwargs'])
    assert rebuilt.task_dropout_rate == 0.41
    assert rebuilt.attention_dropout_rate == 0.31
    if encoder == 'gte':
        assert rebuilt.rope_theta == 1234.0 and rebuilt.rope_scaling_factor == 2.5


def test_checkpoint_binding_rejects_changed_model_or_normalizer_before_assign(trainer, tmp_path):
    from gammagl.models import GraphBERT
    model = GraphBERT(12, 1, hidden_size=4, num_hidden_layers=1,
                      num_attention_heads=1, intermediate_size=8, max_position_embeddings=32)
    model.artifact_fingerprint = 'original-model-normalizer-bundle'
    checkpoint = tmp_path / ('best.npz' if tlx.BACKEND == 'tensorflow' else 'best.pt')
    trainer.save_checkpoint(tlx, checkpoint, model, 1, 0.5)
    model.artifact_fingerprint = 'different-model-normalizer-bundle'
    before = [tlx.convert_to_numpy(w).copy() for w in model.all_weights]
    with pytest.raises(ValueError, match='artifact fingerprint'):
        trainer.restore_checkpoint(tlx, checkpoint, model, 'cpu')
    for weight, original in zip(model.all_weights, before):
        np.testing.assert_array_equal(tlx.convert_to_numpy(weight), original)


def test_artifact_write_failure_stops_before_training_and_keeps_summary_incomplete(
        trainer, monkeypatch, tmp_path):
    from examples.graph_tokenizer import artifacts
    original = artifacts.atomic_write

    def write(path, content):
        if Path(path).name == 'run_manifest.json':
            raise OSError('simulated artifact disk failure')
        return original(path, content)

    monkeypatch.setattr(artifacts, 'atomic_write', write)
    monkeypatch.setattr(trainer, 'run_mlm', lambda *_a: pytest.fail('training started without its manifest'))
    with pytest.raises(OSError, match='artifact disk failure'):
        trainer.main(['--smoke', '--dataset', 'aqsol', '--device', 'cpu', '--seed', '42',
                      '--output-dir', str(tmp_path)])
    group = tmp_path / 'aqsol/bert'
    assert json.loads((group / 'summary.json').read_text())['complete'] is False
    assert not (group / 'run_01_seed_42/summary.json').exists()
    assert not list(group.rglob('best_finetune.*'))


def test_legacy_run_requires_explicit_original_assets(tmp_path):
    from examples.graph_tokenizer.artifacts import load_run
    with pytest.raises(ValueError, match='Legacy.*paired assets'):
        load_run(tmp_path, device='cpu')


def test_interrupted_preprocessing_export_can_retry_without_publishing_partial_bundle(
        trainer, monkeypatch, tmp_path):
    from examples.graph_tokenizer import artifacts
    original = artifacts.atomic_write

    def fail_normalizer(path, content):
        if Path(path).name == 'normalizer.json':
            raise OSError('interrupted preprocessing export')
        return original(path, content)

    argv = ['--smoke', '--dataset', 'aqsol', '--device', 'cpu', '--seed', '42',
            '--output-dir', str(tmp_path)]
    monkeypatch.setattr(artifacts, 'atomic_write', fail_normalizer)
    with pytest.raises(OSError, match='interrupted preprocessing'):
        trainer.main(argv)
    preprocessing = tmp_path / 'aqsol/bert/preprocessing'
    assert not list(preprocessing.glob('*/tokenizer.pkl')), 'incomplete bundle was published'
    monkeypatch.setattr(artifacts, 'atomic_write', original)
    assert trainer.main(argv)['complete'] is True
