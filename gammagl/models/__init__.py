from .gcn import GCNModel
from .gat import GATModel
from .heat import HEAT
from .sgc import SGCModel
from .gatv2 import GATV2Model
from .gaan import GaANModel
from .graphsage import GraphSAGE_Full_Model, GraphSAGE_Sample_Model
from .gcnii import GCNIIModel
from .appnp import APPNPModel
from .gin import GINModel
from .rgcn import RGCN
from .compgcn import CompGCN
from .agnn import AGNNModel
from .jknet import JKNet
from .han import HAN
from .dgi import DGIModel
from .grace import GraceModel
from .grade import GRADE
from .chebnet import ChebNetModel
from .simplehgn import SimpleHGNModel
from .fagcn import FAGCNModel
from .gprgnn import GPRGNNModel
from .dgcnn import DGCNNModel
from .seal import DGCNN
from .hgt import HGTModel
from .film import FILMModel
from .pna import PNAModel
from .mixhop import MixHopModel
from .hcha import HCHA
from .hardgat import HardGATModel
from .mlp import MLP
from .graphgan_generator import Generator
from .graphgan_discriminator import Discriminator
from .graphgan import GraphGAN
from .vgae import VGAEModel, GAEModel,InnerProductDecoder
from .gen import GEstimationN
from .skipgram import SkipGramModel
from .deepwalk import DeepWalkModel
from .node2vec import Node2vecModel
from .hpn import HPN
from .gmm import GMMModel
from .herec import HERec
from .metapath2vec import MetaPath2Vec
from .iehgcn import ieHGCNModel
from .tadw import TADWModel
from .mgnni import MGNNI_m_MLP, MGNNI_m_att
from .magcl import NewGrace
from .cagcn import CAGCNModel
from .cogsl import CoGSLModel
from .ggd import GGDModel
from .specformer import Specformer, SpecLayer
from .grace_pot import Grace_POT_Encoder, Grace_POT_Model
from .sfgcn import SFGCNModel
from .grace_spco import Grace_Spco_Encoder, Grace_Spco_Model
from .graphormer import Graphormer
from .sp2gcl import SpaSpeNode, Encoder, EigenMLP
from .heco import HeCo
from .fusedgat import FusedGATModel
from .hid_net import Hid_net
from .gnnlfhf import GNNLFHFModel
from .dhn import DHNModel
from .dna import DNAModel
from .dfad import DFADModel, DFADGenerator
from .fatragnn import FatraGNNModel, Graph_Editer
from .rohehan import RoheHAN
from .gcil import GCILModel, LogReg
from .sgformer import SGFormerModel
from .adagad import PreModel, ReModel
from .dyfss import MoeSSL,VGAE,Discriminator,InnerProductDecoder
from .egt import EGTModel

__all__ = [
    'HeCo',
    'VGAE',
    'GCNModel',
    'GATModel',
    'GaANModel',
    'SGCModel',
    'GATV2Model',
    'GraphSAGE_Full_Model',
    'GraphSAGE_Sample_Model',
    'GCNIIModel',
    'APPNPModel',
    'GINModel',
    'RGCN',
    'CompGCN',
    'AGNNModel',
    'JKNet',
    'HAN',
    'DGIModel',
    'GraceModel',
    'GRADE',
    'ChebNetModel',
    'SimpleHGNModel',
    'FAGCNModel',
    'GPRGNNModel',
    'DGCNN',
    'MixHopModel',
    'HCHA',
    'HGTModel',
    'PNAModel',
    'HardGATModel',
    # 'GraphGAN',
    # 'Generator',
    'Discriminator',
    'DGCNNModel',
    'FILMModel',
    'GEstimationN',
    'SkipGramModel',
    'DeepWalkModel',
    'Node2vecModel',
    'VGAEModel',
    'GAEModel',
    'HPN',
    'GMMModel',
    'HERec',
    'MetaPath2Vec'
    'ieHGCNModel',
    'TADWModel',
    'MGNNI_m_MLP',
    'MGNNI_m_att',
    'NewGrace',
    'CAGCNModel',
    'CoGSLModel',
    'GGDModel',
    'Specformer',
    'SFGCNModel',
    'Graphormer',
    'Encoder',
    'EigenMLP',
    'SpaSpeNode',
    'FusedGATModel',
    'hid_net',
    'HEAT',
    'GNNLFHFModel',
    'DHNModel',
    'DNAModel',
    'DFADModel',
    'DFADGenerator',
    'FatraGNNModel',
    'RoheHAN',
    'GCILModel',
    'LogReg',
    'sgformer',
    'PreModel',
    'ReModel'
    'MoeSSL',
    'VGAE',
    'Discriminator',
    'InnerProductDecoder'，
    'EGTModel',
]

classes = __all__
