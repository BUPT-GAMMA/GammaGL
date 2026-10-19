"""Device policy checks without acquiring any GPU on the shared machine."""
from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys

import pytest


class TensorFlowConfig:
    def __init__(self, initialized=False, count=2):
        self.physical = [SimpleNamespace(name=f'physical_gpu_{i}') for i in range(count)]
        self.visible = list(self.physical)
        self.growth = {}
        self.initialized = initialized
        self.experimental = self

    def list_physical_devices(self, kind):
        assert kind == 'GPU'
        return self.physical

    def get_visible_devices(self, kind):
        assert kind == 'GPU'
        return self.visible

    def set_visible_devices(self, devices, kind):
        assert kind == 'GPU'
        if self.initialized and devices != self.visible:
            raise RuntimeError('runtime already initialized')
        self.visible = list(devices)

    def get_memory_growth(self, device):
        return self.growth.get(device.name)

    def set_memory_growth(self, device, enabled):
        if self.initialized:
            raise RuntimeError('runtime already initialized')
        self.growth[device.name] = enabled


def test_tensorflow_cpu_hides_every_gpu():
    from examples.graph_tokenizer.runtime import configure_device
    config = TensorFlowConfig()
    result = configure_device('tensorflow', 'cpu', tf_module=SimpleNamespace(config=config))
    assert config.visible == []
    assert result['resolved_device'] == 'cpu'


def test_tensorflow_cuda_index_selects_requested_visible_gpu_idempotently():
    from examples.graph_tokenizer.runtime import configure_device
    config = TensorFlowConfig()
    tf = SimpleNamespace(config=config)
    result = configure_device('tensorflow', 'cuda:1', tf_module=tf)
    assert [d.name for d in config.visible] == ['physical_gpu_1']
    assert config.growth == {'physical_gpu_1': True}
    assert result['requested_device'] == 'cuda:1'
    assert result['resolved_device'] == 'cuda:0'
    config.initialized = True
    assert configure_device('tensorflow', 'cuda:1', tf_module=tf) == result


@pytest.mark.parametrize('device', ['cpu', 'cuda:1'])
def test_tensorflow_initialized_runtime_rejects_device_change(device):
    from examples.graph_tokenizer.runtime import configure_device
    tf = SimpleNamespace(config=TensorFlowConfig(initialized=True))
    with pytest.raises(RuntimeError, match='before.*TensorFlow|TensorFlow.*initialized'):
        configure_device('tensorflow', device, tf_module=tf)


@pytest.mark.parametrize('device', ['cuda:2', 'cuda'])
def test_tensorflow_missing_device_fails_without_cpu_fallback(device):
    from examples.graph_tokenizer.runtime import configure_device
    count = 2 if device == 'cuda:2' else 0
    with pytest.raises(ValueError, match='GPU|device'):
        configure_device('tensorflow', device,
                         tf_module=SimpleNamespace(config=TensorFlowConfig(count=count)))


@pytest.mark.parametrize('device', ['cpu:1', 'cuda:-1', 'gpu', 'cuda:abc', 'cuda:0:1'])
def test_invalid_device_is_rejected_before_any_framework_operation(device):
    from examples.graph_tokenizer.runtime import configure_device
    with pytest.raises(ValueError, match='device'):
        configure_device('tensorflow', device, tf_module=SimpleNamespace())


def test_tensorflow_device_is_selected_before_seed_and_data(monkeypatch, tmp_path):
    from examples.graph_tokenizer import graph_tokenizer_trainer as trainer
    calls = []

    def configure(backend, device):
        calls.append('device')
        return {'requested_device': device, 'resolved_device': device, 'backend': backend}

    def seed(*_args):
        assert calls == ['device']
        calls.append('seed')

    def data(*_args):
        assert calls == ['device', 'seed']
        raise LookupError('stop before model creation')

    monkeypatch.setattr(trainer, 'configure_device', configure, raising=False)
    monkeypatch.setattr(trainer, 'set_seed', seed)
    monkeypatch.setattr(trainer, 'load_or_prepare_preprocessing', data)
    args = trainer.build_parser().parse_args(['--device', 'cpu', '--output-dir', str(tmp_path)])
    with pytest.raises(LookupError, match='stop before'):
        trainer.run_single_experiment(args, trainer.PAPER_CONFIGS[('aqsol', 'bert')], 42)


@pytest.mark.skipif(os.environ.get('TL_BACKEND') != 'tensorflow', reason='TensorFlow CLI bootstrap')
def test_direct_tensorflow_cli_selects_cpu_before_tlx_import(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = '''
import runpy, sys
from examples.graph_tokenizer import runtime
original = runtime.configure_device
calls = []
def checked(backend, device):
    if not calls:
        assert 'tensorlayerx' not in sys.modules
    calls.append(device)
    return original(backend, device)
runtime.configure_device = checked
sys.argv = [sys.argv[1], '--smoke', '--dataset', 'aqsol', '--seed', '42',
            '--device', 'cpu', '--output-dir', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
assert calls == ['cpu', 'cpu']
import tensorflow as tf
assert 'CPU' in tf.constant(1).device
'''
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(root)}
    result = subprocess.run([sys.executable, '-c', script,
                             str(root / 'examples/graph_tokenizer/graph_tokenizer_trainer.py'),
                             str(tmp_path)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
