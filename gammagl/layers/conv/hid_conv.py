import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
import tensorlayerx as tlx
from gammagl.mpops import *
from gammagl.utils.num_nodes import maybe_num_nodes


def cal_g_gradient(edge_index, x, edge_weight=None, sigma1=0.5, sigma2=0.5, num_nodes=None,
             dtype=None):
    row, col = edge_index[0], edge_index[1]
    ones = tlx.ones([edge_index[0].shape[0]])
    if dtype is not None:
        ones = tlx.cast(ones, dtype)
    if num_nodes is None:
        num_nodes = int(1 + tlx.reduce_max(edge_index[0]))
    deg = unsorted_segment_sum(ones, col, num_segments=num_nodes)
    deg_inv = tlx.pow(deg+1e-8, -1)
    deg_in_row = tlx.reshape(tlx.gather(deg_inv, row), (-1,1))
    x_row = tlx.gather(x, row)
    x_col = tlx.gather(x, col)
    gra = deg_in_row * (x_col - x_row)
    avg_gra = unsorted_segment_sum(gra, row, num_segments=x.shape[0])
    dx = x_row - x_col
    norms_dx = tlx.reduce_sum(tlx.square(dx), axis=1)
    norms_dx = tlx.sqrt(norms_dx)
    s = norms_dx
    s = tlx.exp(- (s * s) / (2 * sigma1 * sigma2))
    r = unsorted_segment_sum(tlx.reshape(s,(-1,1)), row,num_segments=x.shape[0])
    r_row = tlx.gather(r, row)
    coe = tlx.reshape(s, (-1,1)) / (r_row + 1e-6)
    avg_gra_row = tlx.gather(avg_gra, row)
    result = unsorted_segment_sum(avg_gra_row * coe, col, num_segments=x.shape[0])
    return result

class Hid_conv(MessagePassing):
    r'''The proposed high-order graph diffusion equation is given by:

    .. math::
        \frac{\partial x(t)_i}{\partial t} = 
        \alpha(x(0)_i - x(t)_i) + 
        \beta \text{div}(f(\nabla x(t)_{ij})) + 
        \gamma(\nabla x(t)_j),

    where \( \alpha \), \( \beta \), and \( \gamma \) are parameters of the model. 
    This equation integrates the high-order diffusion process by considering the influence of both first-order and second-order neighbors in the graph. 
    The iteration step based on this equation is formulated as:

    .. math::
        x(t+\Delta t)_i = \\alpha \Delta t x(0)_i + 
        (1 - \alpha \Delta t)x(t)_i + \beta \Delta t \text{div}(f(\nabla x(t)_i)) + 
            \beta \gamma \Delta t \text{div}((\nabla x(t))_j),

    which represents the diffusion-based message passing scheme (DMP) of the High-order Graph Diffusion Network (HiD-Net). 
    This scheme leverages the information from two-hop neighbors, offering two main advantages: 
    it captures the local environment around a node, enhancing the robustness of the model against abnormal features within one-hop neighbors; 
    and it utilizes the monophily property of two-hop neighbors, which provides a stronger correlation with labels 
    and thus enables better predictions even in the presence of heterophily within one-hop neighbors.

    Parameters
    ----------
    alpha: float
    beta: float
    gamma: float
    sigma1: float
    sigma2: float

    '''
    def __init__(self,
                 alpha,
                 beta,
                 gamma,
                 sigma1,
                 sigma2):
        super().__init__()
        self.alpha = alpha
        self.beta =  beta
        self.gamma = gamma
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.reset_parameters()

    def reset_parameters(self):
        
        self._cached_edge_index = None
        self._cached_adj_t = None
    
    def forward(self, x, origin, edge_index, edge_weight, ei_no_loops, ew_no_loops, num_nodes=None):
        if num_nodes == None:
             num_nodes = maybe_num_nodes(edge_index, num_nodes)
    
        ew2 = tlx.reshape(ew_no_loops, (-1, 1))

        g = cal_g_gradient(edge_index=ei_no_loops, x=x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2, dtype=None)

        Ax = self.propagate(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
        Gx = self.propagate(g, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)

        x = self.alpha * origin + (1 - self.alpha - self.beta) * x \
            + self.beta * Ax \
            + self.beta * self.gamma * Gx
            
        return x
