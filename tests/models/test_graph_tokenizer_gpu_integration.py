"""Explicit opt-in GPU checks; run only after reserving two unused GPUs."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('device', ['cpu', 'cuda:1'])
def test_gpu_device_selection_training_and_restore(tmp_path, device):
    configured = os.environ.get('GAMMAGL_GRAPH_TOKENIZER_TEST_GPU_UUIDS')
    if not configured:
        pytest.skip('set two reserved GPU UUIDs to enable hardware integration')
    devices = configured.split(',')
    if len(devices) != 2 or len(set(devices)) != 2 or not all(value.startswith('GPU-') for value in devices):
        raise ValueError('Hardware integration requires two distinct GPU UUIDs')
    script = '''
import sys, numpy as np
from examples.graph_tokenizer.runtime import configure_device
backend, device, output = sys.argv[1:]
runtime = configure_device(backend, device)
assert configure_device(backend, device) == runtime
from examples.graph_tokenizer import graph_tokenizer_trainer as trainer
import tensorlayerx as tlx
if backend == 'tensorflow':
    import tensorflow as tf
    assert tf.test.is_built_with_cuda(), 'GPU-enabled TensorFlow build required'
    probe = tf.constant(1.)
    assert ('GPU' if device != 'cpu' else 'CPU') in probe.device
    assert len(tf.config.get_visible_devices('GPU')) == (0 if device == 'cpu' else 1)
    assert runtime['resolved_device'] == ('cpu' if device == 'cpu' else 'cuda:0')
else:
    import torch
    if device != 'cpu':
        torch.cuda.set_per_process_memory_fraction(0.1, device=device)
    probe = torch.ones(1, device=device)
    assert str(probe.device) == device
captured = {}
original = trainer.make_model
def capture(*args, **kwargs):
    model = original(*args, **kwargs)
    captured['model'] = model
    return model
trainer.make_model = capture
trainer.main(['--smoke', '--dataset', 'aqsol', '--seed', '42',
              '--device', device, '--output-dir', output])
model = captured['model']
graph = trainer.synthetic_splits(trainer.resolve_dataset('aqsol'))['test'][0]
from examples.graph_tokenizer.artifacts import load_run
run = load_run(output + '/aqsol/bert/run_01_seed_42', device=device)
ids = tlx.convert_to_tensor([run['tokenizer'].encode_graph(graph).input_ids], dtype=tlx.int64)
mask = tlx.ones_like(ids)
if backend == 'torch':
    ids, mask = ids.to(device), mask.to(device)
    assert str(next(model.parameters()).device) == device
else:
    assert ('GPU' if device != 'cpu' else 'CPU') in model.all_weights[0].device
expected = tlx.convert_to_numpy(model(ids, mask, task='supervised'))
actual = tlx.convert_to_numpy(run['model'](ids, mask, task='supervised'))
np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
if backend == 'tensorflow':
    try:
        configure_device(backend, 'cuda:1' if device == 'cpu' else 'cpu')
    except RuntimeError:
        pass
    else:
        raise AssertionError('Initialized TensorFlow must reject an incompatible device change')
try:
    configure_device(backend, 'cuda:2')
except ValueError:
    pass
else:
    raise AssertionError('Unavailable GPU must be rejected')
'''
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(devices),
           'PYTHONPATH': os.environ.get('PYTHONPATH', str(Path(__file__).resolve().parents[2]))}
    result = subprocess.run([sys.executable, '-c', script, os.environ['TL_BACKEND'],
                             device, str(tmp_path)], env=env, capture_output=True,
                            text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
