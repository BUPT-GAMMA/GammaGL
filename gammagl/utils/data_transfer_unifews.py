"""Deprecated: Data conversion helpers moved to examples/unifews/convert_data.py

The classes in this module have been moved to examples/unifews/convert_data.py
with configurable dataset paths. Use that module instead.
"""

import warnings

warnings.warn(
    "gammagl.utils.data_transfer_unifews is deprecated. "
    "Use examples.unifews.convert_data instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['DataProcess_OGB', 'DataProcess_PyGFlickr', 'DataProcess_PyG']  # noqa: F822


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from examples.unifews import convert_data
    return getattr(convert_data, name)
