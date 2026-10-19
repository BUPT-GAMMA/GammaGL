"""Backend device configuration before seeds, tensors or model construction."""
import os
import re


def configure_device(backend, device, *, tf_module=None):
    """Resolve cpu/cuda[:N]; TensorFlow indices refer to OS-visible GPUs.

    TensorFlow remaps the selected physical device to logical GPU 0. Repeating
    an identical choice is safe; an initialized incompatible runtime is an
    error rather than a silent fallback. tf_module permits hardware-free tests.
    """
    if device != 'cpu' and not re.fullmatch(r'cuda(?::[0-9]+)?', device):
        raise ValueError(f'Invalid device {device!r}; use cpu, cuda or cuda:N.')
    index = int(device.split(':')[1]) if ':' in device else 0
    result = dict(backend=backend, requested_device=device,
                  resolved_device='cpu' if device == 'cpu' else f'cuda:{index}',
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if backend == 'tensorflow':
        if tf_module is None:
            import tensorflow as tf_module
        config = tf_module.config
        physical = config.list_physical_devices('GPU')
        if device != 'cpu' and index >= len(physical):
            raise ValueError(f'Requested device {device}: only {len(physical)} GPU devices are available.')
        selected = [] if device == 'cpu' else [physical[index]]
        try:
            if config.get_visible_devices('GPU') != selected:
                config.set_visible_devices(selected, 'GPU')
            for gpu in selected:
                if config.experimental.get_memory_growth(gpu) is not True:
                    config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            raise RuntimeError(
                'Configure the requested device before creating TensorFlow tensors; '
                'the TensorFlow runtime is already initialized. Start a fresh process.') from error
        result.update(visible_devices=[gpu.name for gpu in selected],
                      resolved_device='cpu' if not selected else 'cuda:0')
    elif backend == 'torch':
        if device != 'cpu':
            import torch
            if index >= torch.cuda.device_count():
                raise ValueError(f'Requested device {device}: GPU index is unavailable.')
    else:
        raise RuntimeError('GraphTokenizer training supports torch and TensorFlow eager.')
    return result
