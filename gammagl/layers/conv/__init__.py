from .heat_conv import HEATlayer
from .message_passing import MessagePassing
from .gcn_conv import GCNConv
from .gat_conv import GATConv
from .gaan_conv import GaANConv
from .sgc_conv import SGConv
from .sage_conv import SAGEConv
from .gatv2_conv import GATV2Conv
from .gcnii_conv import GCNIIConv
from .appnp_conv import APPNPConv
from .rgcn_conv import RGCNConv
from .compgcn_conv import CompConv
from .agnn_conv import AGNNConv
from .JumpingKnowledge import JumpingKnowledge
from .han_conv import HANConv
from .cheb_conv import ChebConv
from .simplehgn_conv import SimpleHGNConv
from .fagcn_conv import FAGCNConv
from .gpr_conv import GPRConv
from .gin_conv import GINConv
from .hgt_conv import HGTConv
from .mixhop_conv import MixHopConv
from .hcha_conv import HypergraphConv
from .hardgat_conv import HardGATConv
from .pna_conv import PNAConv
from .film_conv import FILMConv
from .edgeconv import EdgeConv
from .hpn_conv import HPNConv
from .hetero_wrapper import HeteroConv
from .gmm_conv import GMMConv
from .iehgcn_conv import ieHGCNConv
from .mgnni_m_iter import MGNNI_m_iter
from .magcl_conv import MAGCLConv
from .fusedgat_conv import FusedGATConv
from .hid_conv import Hid_conv
from .dhn_conv import DHNConv
from .dna_conv import DNAConv
from .rohehan_conv import RoheHANConv
__all__ = [
    'MessagePassing',
    'GCNConv',
    'GATConv',
    'GaANConv',
    'SGConv',
    'SAGEConv',
    'GATV2Conv',
    'GCNIIConv',
    'APPNPConv',
    'RGCNConv',
    'AGNNConv',
    'JumpingKnowledge',
    'HANConv',
    'ChebConv',
    'HeteroConv',
    'SimpleHGNConv',
    'FAGCNConv',
    'GPRConv',
    'HGTConv',
    'MixHopConv',
    'HypergraphConv',
    'HardGATConv',
    'PNAConv',
    'FILMConv',
    'CompConv',
    'EdgeConv',
    'HPNConv',
    'GINConv',
    'GMMConv',
    'ieHGCNConv',
    'MGNNI_m_iter',
    'MAGCLConv',
    'FusedGATConv',
    'Hid_conv',
    'HEATlayer',
    'DHNConv',
    'DNAConv',
    'RoheHANConv',
    'EuclideanEncoder',
    'ManifoldEncoder',
    'ConstCurveLinear',
    'ConstCurveAgg',
]

classes = __all__

# Lazy imports for RGT layers (to avoid circular dependency issues)
def __getattr__(name):
    if name in ('EuclideanEncoder', 'ManifoldEncoder', 'ConstCurveLinear', 'ConstCurveAgg'):
        from .rgt_layers import EuclideanEncoder, ManifoldEncoder, ConstCurveLinear, ConstCurveAgg
        return {'EuclideanEncoder': EuclideanEncoder, 'ManifoldEncoder': ManifoldEncoder,
                'ConstCurveLinear': ConstCurveLinear, 'ConstCurveAgg': ConstCurveAgg}[name]
    if name in ('Lorentz', 'Sphere', 'Euclidean', 'ProductSpace'):
        if name == 'Lorentz':
            from .manifold_lorentz import Lorentz
            return Lorentz
        if name == 'Sphere':
            from .manifold_sphere import Sphere
            return Sphere
        if name == 'Euclidean':
            from .manifold_euclidean import Euclidean
            return Euclidean
        if name == 'ProductSpace':
            from .manifold_product import ProductSpace
            return ProductSpace
    if name in ('VectorQuantize_E', 'VectorQuantize_R'):
        if name == 'VectorQuantize_E':
            from .vq_euclidean import VectorQuantize_E
            return VectorQuantize_E
        if name == 'VectorQuantize_R':
            from .vq_riemann import VectorQuantize_R
            return VectorQuantize_R
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
