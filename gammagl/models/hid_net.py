import tensorlayerx as tlx
from gammagl.layers.conv import Hid_conv
from gammagl.utils.loop import remove_self_loops, add_self_loops,contains_self_loops
from gammagl.mpops import *
from gammagl.utils.norm import calc_gcn_norm


class Hid_net(tlx.nn.Module):
    r"""High-Order Graph Diffusion Network (HiD-Net) proposed in `"A Generalized Neural Diffusion Framework on Graphs"
        <https://arxiv.org/abs/2312.08616>`_ paper.
        
        Parameters
        ----------
        in_feats: int
            input feature dimension.
        n_hidden: int
            hidden dimension.
        n_classes: int
            number of classes.
        alpha: float
        beta: float
        gamma: float
        bias: bool 
            add bias or not.
        drop: bool
            drop or not.
        dropout: float
            the value of dropout.
        sigma1: float
        sigma2: float
        

    """
    def __init__(self, in_feats,
                 n_hidden,
                 n_classes,
                 k,
                 alpha,
                 beta,
                 gamma,
                 bias,
                 normalize,
                 drop,
                 dropout,
                 sigma1,
                 sigma2):
        super(Hid_net, self).__init__()
        initor=tlx.initializers.xavier_normal()
        self.lin1 = tlx.layers.Linear(in_features=in_feats, out_features=n_hidden, W_init=initor,b_init=None)
        self.lin2 = tlx.layers.Linear(in_features=n_hidden, out_features=n_classes, W_init=initor,b_init=None)
        self.relu = tlx.ReLU()
        self.normalize=normalize
        self.drop=drop
        self.Dropout = tlx.layers.Dropout(dropout)
        if bias:
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1,self.out_channels), init=initor)
        self.convs = tlx.nn.ModuleList()
        for i in range (k):
            self.convs.append(Hid_conv(
                    alpha,
                    beta,
                    gamma,
                    sigma1,
                    sigma2))
            
    def forward(self, x, edge_index, edge_weight=None,num_nodes=None):
        if self.normalize:
            edgei = edge_index
            edgew = edge_weight

            if edge_weight is None:
                edge_weight = tlx.ones(shape=(edge_index.shape[1], 1)) 
            
            if contains_self_loops(edge_index):
                edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
                edge_index,edge_weight= add_self_loops(edge_index, edge_weight)
            else:
                edge_index,edge_weight= add_self_loops(edge_index, edge_weight)    
            

            edge_weight=calc_gcn_norm(edge_index,num_nodes,edge_weight)

            edge_index_without_selfloops=edgei
            edge_weight_without_selfloops=calc_gcn_norm(edgei,num_nodes,edgew)
            

        if self.drop == 'True':
            x = self.Dropout(x)

        x = self.lin1(x)
        x = self.relu(x)
        x = self.Dropout(x)
        x = self.lin2(x)
        origin = x
        for l, layer in enumerate(self.convs):
            x = layer(x,origin, edge_index,edge_weight,edge_index_without_selfloops,edge_weight_without_selfloops,num_nodes)
        return x