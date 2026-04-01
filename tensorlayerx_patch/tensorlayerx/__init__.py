# Proxy module that forwards to real tensorlayerx
import sys
import os

# Remove our patch from sys.modules and path temporarily
if 'tensorlayerx' in sys.modules:
    del sys.modules['tensorlayerx']
    
original_path = sys.path[:]
patch_dir = os.path.dirname(os.path.dirname(__file__))
if patch_dir in sys.path:
    sys.path.remove(patch_dir)

try:
    # Import the real tensorlayerx
    import tensorlayerx as _real_tlx
    
    # Copy all attributes from real tensorlayerx to this module
    for attr in dir(_real_tlx):
        if not attr.startswith('_'):
            globals()[attr] = getattr(_real_tlx, attr)
    
    # Also copy special attributes
    __version__ = getattr(_real_tlx, '__version__', '0.5.8')
    __file__ = _real_tlx.__file__
    __path__ = getattr(_real_tlx, '__path__', [])
    
finally:
    # Restore path
    sys.path[:] = original_path
