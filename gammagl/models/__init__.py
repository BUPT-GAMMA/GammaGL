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
from .dgcnn import DGCNNModel
from .seal import DGCNN
from .hgt import HGTModel
from .film import FILMModel
from .pna import PNAModel
from .mixhop import MIXHOPModel
from .hcha import HCHA
from .hardgat import HardGATModel
from .mlp import MLP
from .graphgan_generator import Generator
from .graphgan_discriminator import Discriminator
from .graphgan import GraphGAN
from .vgae import VGAEModel, GAEModel
from .gen import GEstimationN
from .deepwalk import DeepWalkModel
from .node2vec import Node2vecModel
from .hpn import HPN


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
    'SimpleHGNModel',
    'FAGCNModel',
    'GPRGNNModel',
    'DGCNN',
    'MIXHOPModel',
    'HCHA',
    'HGTModel',
    'PNAModel',
    'HardGATModel',
    # 'GraphGAN',
    # 'Generator',
    # 'Discriminator',
    'DGCNNModel',
    'FILMModel',
    'GEstimationN',
    'DeepWalkModel',
    'Node2vecModel',
    'VGAEModel',
    'GAEModel',
    'HPN'
    
]

classes = __all__
