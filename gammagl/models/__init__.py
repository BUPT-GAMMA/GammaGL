from .gcn import GCNModel
from .gat import GATModel
from .sgc import SGCModel
from .gatv2 import GATV2Model
from .graphsage import GraphSAGE_Full_Model, GraphSAGE_Sample_Model
from .gcnii import GCNIIModel
from .appnp import APPNPModel
from .gin import GINModel
from .rgcn import RGCN
from .agnn import AGNNModel
from .jknet import JKNet
from .han import HAN
from .dgi import DGIModel
from .grace import GraceModel
from .chebnet import ChebNetModel
from .simplehgn import SimpleHGNModel
from .fagcn import FAGCNModel
from .gprgnn import GPRGNNModel

__all__ = [
    'GCNModel',
    'GATModel',
    'SGCModel',
    'GATV2Model',
    'GraphSAGE_Full_Model',
    'GraphSAGE_Sample_Model',
    'GCNIIModel',
    'APPNPModel',
    'GINModel',
    'RGCN',
    'AGNNModel',
    'JKNet',
    'HAN',
    'DGIModel',
    'GraceModel',
    'ChebNetModel',
    'SimpleHGNModel'
    'FAGCNModel',
    'GPRGNNModel'
]

classes = __all__
