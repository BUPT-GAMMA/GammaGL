import tensorlayerx as tlx
import pytest
from gammagl.utils.device import set_device


def test_set_device_gpu():
    if tlx.is_gpu_available():
        try:
            set_device(0)
            assert tlx.get_device() == "GPU:0", "Failed to set GPU device"
        except Exception as e:
            pytest.fail(f"Setting GPU device raised an exception: {e}")

def test_set_device_cpu():
    try:
        set_device(-1)
        assert tlx.get_device() == "CPU:0", "Failed to set CPU device"
    except Exception as e:
        pytest.fail(f"Setting CPU device raised an exception: {e}")
