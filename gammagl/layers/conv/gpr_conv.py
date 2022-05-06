import tensorlayerx as tlx
import numpy as np
from gammagl.layers.conv import MessagePassing
# from tensorlayerx.nn.layers.utils import (get_variable_with_initializer, random_normal)
from tensorlayerx.nn.initializers import Initializer

class GPRConv(MessagePassing):
    r"""The graph propagation oeprator from the `"Adaptive 
    Universal Generalized PageRank Graph Neural Network"
    <https://arxiv.org/abs/2006.07988.pdf>`_ paper

    .. math::
        \mathbf{H}^{(k)} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{H}^{(k-1)} \\
        \mathbf{Z} = \sum\limits_{k=0}^{k=K}\gamma_k\mathbf{H}^{(k)}.

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.
    :math:`\gamma_{k}` its learnt weights.

    Parameters:
        k: steps to propagate.
        alpha: assgin initial value to learnt weights, used in concert with Init.
        Init: initialization method(SGC, PPR, NPPR, Random, WS).
        add_bias: If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """


    def __init__(self, K, alpha, Init='PPR', Gamma=None, bias=True, **kwargs):
        super(GPRConv, self).__init__(**kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma
        #########
        ### If use tlx.Variable() to set gamma as trainable,
        ### when backend is "tensorflow", though gamma is require_gradient, model would not add gamma to model.trainable_parameters
        ### then we must use net.all_parameters() to abstract parameters gamma (not elegant!)
        #########
        self.temp = tlx.Variable(tlx.convert_to_tensor(TEMP, dtype='float32'), name='/gamma', trainable=True)


    def reset_parameters(self):
        self.temp = tlx.zeros_like(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        # edge_index, norm = gcn_norm(
        #     edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    # def message(self, x_j, norm):
    #     return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
