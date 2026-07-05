"""DeFoG-specific graph generation datasets.

These datasets are designed for discrete flow matching graph generation
and contain heavy preprocessing (dense adjacency matrices, atom/bond
distribution statistics, eigenvalue caching, etc.) that is specific to
the DeFoG training pipeline.

They are intentionally kept in ``examples/defog/datasets/`` rather than
``gammagl/datasets/`` to avoid polluting the core GammaGL package with
optional heavy dependencies (RDKit, graph-tool, etc.).
"""

from .spectre_dataset import (
    PlanarGraphDataset,
    TreeGraphDataset,
    SBMGraphDataset,
    Comm20GraphDataset,
)
from .qm9_dataset import QM9Gen
from .moses_dataset import MOSESDataset
from .guacamol_dataset import GuacaMolDataset
from .zinc250k_dataset import ZINC250kGen
from .tls_dataset import TLSGraphDataset

__all__ = [
    'PlanarGraphDataset',
    'TreeGraphDataset',
    'SBMGraphDataset',
    'Comm20GraphDataset',
    'QM9Gen',
    'MOSESDataset',
    'GuacaMolDataset',
    'ZINC250kGen',
    'TLSGraphDataset',
]
