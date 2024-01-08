import tensorlayerx as tlx
import gammagl.mpops as mpops
from gammagl.layers.conv import GPRConv




class GPRGNNModel(tlx.nn.Module):
    r"""Graph Convolutional Network proposed in `"Adaptive 
        Universal Generalized PageRank Graph Neural Network"
        <https://arxiv.org/abs/2006.07988.pdf>`_ paper.
        
        Parameters
        -----------
        feature_dim:
            input feature dim.
        hidden_dim: 
            hidden feature dim.
        num_class: 
            output feature dim.
        drop_rate: 
            drop rate for dropout layer.
        K: 
            K steps to propagate.
        Init: 
            initialization method.
        alpha: 
            help to initial the value of gamma.
        dprate: 
            special dropout for GPR-GNN (before starting GPR propagation) to dropout some nodes.
        Gamma: 
            Only used when Init == "WS", self-defined initial value of gamma.

    """

    def __init__(self, feature_dim, hidden_dim, num_class, drop_rate, K, Init, alpha, dprate, Gamma=None):
        super(GPRGNNModel, self).__init__()
        self.lin1 = tlx.layers.Linear(out_features=hidden_dim,
                                        in_features=feature_dim,
                                        b_init=None)
        self.lin2 = tlx.layers.Linear(out_features=num_class,
                                        in_features=hidden_dim,
                                        b_init=None)
        # if args.ppnp == 'PPNP':
        #     self.conv = APPNPConv(args.K, args.alpha)
        # elif args.ppnp == 'GPR_prop':
        self.conv = GPRConv(K, alpha, Init, Gamma)
        self.relu = tlx.ReLU()
        self.drop1 = tlx.layers.Dropout(drop_rate)
        self.drop2 = tlx.layers.Dropout(dprate)

        self.Init = Init
        self.dprate = dprate

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self,  x, edge_index, edge_weight, num_nodes):
        x = self.drop1(x)
        x = self.relu(self.lin1(x))
        x = self.drop1(x)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.conv(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
            return x
        else:
            x = self.drop2(x)
            x = self.conv(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
            return x
