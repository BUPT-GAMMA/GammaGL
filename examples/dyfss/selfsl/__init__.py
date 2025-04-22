from .dgi import DGI, DGISample
from .pairdis import PairwiseDistance
from .clustering import Clu
from .partition import Par
from .gnn_encoder import GCN
from .pairsim import PairwiseAttrSim
from .naive_gate import NaiveGate

__all__ = ['DGI', 'GCN', 'PairwiseDistance', 'Clu',
        'Par', 'PairwiseAttrSim', 'DGISample']

