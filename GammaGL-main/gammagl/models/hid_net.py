import tensorlayerx as tlx
from gammagl.layers.conv import Hid_conv
from gammagl.utils.loop import remove_self_loops, add_self_loops, contains_self_loops
from gammagl.mpops import *
from gammagl.utils.norm import calc_gcn_norm


class Hid_net(tlx.nn.Module):
    r"""High-Order Graph Diffusion Network (HiD-Net) proposed in `"A Generalized Neural Diffusion Framework on Graphs"
        <https://arxiv.org/abs/2312.08616>`_ paper.
        
        Parameters
        ----------
        in_feats: int
            input feature dimension.
        hidden_dim: int
            hidden dimension.
        n_classes: int
            number of classes.
        alpha: float
        beta: float
        gamma: float
        add_bias: bool 
            add bias or not.
        drop_rate: float
            dropout rate.
        sigma1: float
        sigma2: float
        name: str, optional
            model name.

    """
    def __init__(self, in_feats,
                 hidden_dim,
                 n_classes,
                 num_layers,
                 alpha,
                 beta,
                 gamma,
                 add_bias,
                 normalize,
                 drop_rate,
                 sigma1,
                 sigma2,
                 name = None):
        super(Hid_net, self).__init__(name=name)
        initor = tlx.initializers.xavier_normal()
        self.lin1 = tlx.layers.Linear(in_features=in_feats, out_features=hidden_dim, W_init=initor, b_init=None)
        self.lin2 = tlx.layers.Linear(in_features=hidden_dim, out_features=n_classes, W_init=initor, b_init=None)
        self.relu = tlx.ReLU()
        self.normalize = normalize
        self.dropout = tlx.layers.Dropout(drop_rate)
        self.add_bias = add_bias
        if add_bias:
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1, n_classes), init=initor)
        self.convs = tlx.nn.ModuleList()
        for i in range (num_layers):
            self.convs.append(Hid_conv(alpha, beta, gamma, sigma1, sigma2))
            
    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        if self.normalize:
            ei_no_loops = edge_index
            ew_no_loops = calc_gcn_norm(edge_index, num_nodes, edge_weight)
            
            if contains_self_loops(edge_index):
                edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
                edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
            else:
                edge_index, edge_weight = add_self_loops(edge_index, edge_weight)    
            edge_weight = calc_gcn_norm(edge_index, num_nodes, edge_weight)

        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        origin = x
        for l, layer in enumerate(self.convs):
            x = layer(x, origin, edge_index, edge_weight, ei_no_loops, ew_no_loops, num_nodes)
        
        if self.add_bias:
            x = self.bias + x
        
        return x
