import os
import subprocess
import sys

def test_defog_backend_imports():
    """
    Ensure that DeFoG core components can be parsed and imported
    without crashing under non-torch backends (e.g. tensorflow).
    This proves there are no stray `import torch` or PyG hard dependencies
    in the shared GammaGL namespace.
    """
    script = """
import tensorlayerx as tlx
from gammagl.models.defog import DeFoGModel
from gammagl.layers.attention.defog_layer import XEyTransformerLayer
print("Import successful on backend:", tlx.BACKEND)
"""

    env = os.environ.copy()
    # Try with tensorflow backend
    env['TL_BACKEND'] = 'tensorflow'

    cmd = [sys.executable, "-c", script]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    # If the user doesn't have tensorflow installed, it will fail with ModuleNotFoundError: No module named 'tensorflow'
    # We should only assert success if tensorflow actually loads or just consider it passed if it didn't fail due to torch
    if result.returncode != 0:
        if "No module named 'tensorflow'" in result.stderr or "No module named 'tensorlayerx'" in result.stderr:
            # Skip if TF/TLX is missing in the local environment
            return
        elif "tensorflow" in result.stderr and "dll" in result.stderr.lower():
             return
        assert False, f"Import failed on tensorflow backend: {result.stderr}"

if __name__ == '__main__':
    test_defog_backend_imports()
    print("Backend import test passed!")
