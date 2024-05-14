import tensorlayerx as tlx

from gammagl.layers.conv import Hid_conv
from gammagl.utils.loop import remove_self_loops, add_self_loops,contains_self_loops
from gammagl.mpops import *
def gcn_norm(edge_index, num_nodes, edge_weight=None,add_self_loop=False):
   
    if edge_weight is None:
        edge_weight = tlx.ones(shape=(edge_index.shape[1], 1))  
    if add_self_loop:
        if contains_self_loops(edge_index):
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
            edge_index,edge_weight= add_self_loops(edge_index, edge_weight)
        else:
            edge_index,edge_weight= add_self_loops(edge_index, edge_weight)
    src, dst = edge_index[0], edge_index[1]


    deg = tlx.reshape(unsorted_segment_sum(edge_weight,src , num_segments=edge_index[0].shape[0]), (-1,))
    deg_inv_sqrt = tlx.pow(deg+1e-8, -0.5)
    weights = tlx.gather(deg_inv_sqrt, src) * tlx.reshape(edge_weight, (-1,)) * tlx.gather(deg_inv_sqrt, dst)
    return edge_index,weights

class hid_net(tlx.nn.Module):
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
        add_self_loops: bool
            add loops or not.
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
                 add_self_loops,
                 drop,
                 dropout,
                 sigma1,
                 sigma2):
        super(hid_net, self).__init__()
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
                    bias,
                    normalize,
                    add_self_loops,
                    drop,
                    dropout,
                    sigma1,
                    sigma2))
            
    def forward(self, x, edge, edge_weight=None,num_nodes=0):
        origin = x
        if self.normalize:
            edgei = edge_index
            edgew = edge_weight

            edge_index,edge_weight = gcn_norm(
                edgei, num_nodes,edgew,add_self_loop=True)

            edge_index2,edge_weight2 = gcn_norm(
                edgei, num_nodes,edgew,add_self_loop=False)

        if self.drop == 'True':
            x = self.Dropout(x)

        x = self.lin1(x)
        x = self.relu(x)
        x = self.Dropout(x)
        x = self.lin2(x)
        h = x
        for l, layer in enumerate(self.convs):
            x = layer(x,origin, edge_index,edge_weight,edge_index2,edge_weight2,num_nodes)
        return x