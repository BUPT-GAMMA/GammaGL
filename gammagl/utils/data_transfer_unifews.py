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

from examples.unifews.convert_data import (
    DataProcess_OGB,
    DataProcess_PyGFlickr,
    DataProcess_PyG,
)
