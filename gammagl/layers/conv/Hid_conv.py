import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
import tensorlayerx as tlx
from gammagl.mpops import *
from gammagl.utils.loop import remove_self_loops, add_self_loops,contains_self_loops

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

def cal_g_gradient(edge_index, x, edge_weight=None, sigma1=0.5, sigma2=0.5, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    row, col = edge_index[0], edge_index[1]
    ones = tlx.ones([edge_index[0].shape[0]])
    if dtype is not None:
        ones = tlx.cast(ones, dtype)
    if num_nodes is None:
        num_nodes = int(1 + tlx.reduce_max(edge_index[0]))
    deg=unsorted_segment_sum(ones, col, num_segments=num_nodes)
    deg_inv = tlx.pow(deg+1e-8, -1)
    deg_in_row=tlx.reshape(tlx.gather(deg_inv,row),(-1,1))
    x_row=tlx.gather(x,row)
    x_col=tlx.gather(x,col)
    gra = deg_in_row * (x_col - x_row)
    avg_gra=unsorted_segment_sum(gra, row, num_segments=x.shape[0])
    dx = x_row-x_col
    norms_dx = tlx.reduce_sum(tlx.square(dx), axis=1)
    norms_dx = tlx.sqrt(norms_dx)
    s = norms_dx
    s = tlx.exp(- (s * s) / (2 * sigma1 * sigma2))
    r=unsorted_segment_sum(tlx.reshape(s,(-1,1)),row,num_segments=x.shape[0])
    r_row=tlx.gather(r,row)
    coe = tlx.reshape(s,(-1,1)) / (r_row + 1e-6)
    avg_gra_row=tlx.gather(avg_gra,row)
    result=unsorted_segment_sum(avg_gra_row * coe, col, num_segments=x.shape[0])
    return result

class Hid_conv(MessagePassing):
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
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.beta =  beta
        self.gamma = gamma
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.drop = drop
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        initor=tlx.initializers.xavier_normal()
        self.lin1 = tlx.layers.Linear(in_features=in_feats, out_features=n_hidden, W_init=initor,b_init=None)
        self.lin2 = tlx.layers.Linear(in_features=n_hidden, out_features=n_classes, W_init=initor,b_init=None)
        self.relu = tlx.ReLU()
        
        self.Dropout = tlx.layers.Dropout(self.dropout)
        if bias:
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1,self.out_channels), init=initor)
        
        self.reset_parameters()

    def reset_parameters(self):
        
        self._cached_edge_index = None
        self._cached_adj_t = None
    
    def forward(self, x, edge_index, edge_weight=None,num_nodes=0):
        
        if self.normalize:
            edgei = edge_index
            edgew = edge_weight

            edge_index,edge_weight = gcn_norm(
                edgei, num_nodes,edgew,add_self_loop=True)

            edge_index2,edge_weight2 = gcn_norm(
                edgei, num_nodes,edgew,add_self_loop=False)

            ew=tlx.reshape(edge_weight,(-1,1))
            ew2=tlx.reshape(edge_weight2,(-1,1))

        if self.drop == 'True':
            x = self.Dropout(x)

        
        x = self.lin1(x)
        x = self.relu(x)
        x = self.Dropout(x)
        x = self.lin2(x)
        h = x

        for k in range(self.k):
            g=cal_g_gradient(edge_index=edge_index2,x=x,edge_weight=ew2,sigma1=self.sigma1,sigma2=self.sigma2,dtype=None)
            
            x1=x
            Ax=x
            Gx=x

            Ax=self.propagate(Ax,edge_index,edge_weight=edge_weight,num_nodes=num_nodes)
            Gx=self.propagate(g,edge_index,edge_weight=edge_weight,num_nodes=num_nodes)

            x = self.alpha * h + (1 - self.alpha - self.beta) * x1 \
                + self.beta * Ax \
                 + self.beta * self.gamma * Gx
            
           
        return x
    def message(self, x, edge_index, edge_weight=None):
        x_j=tlx.gather(x,edge_index[0,:])
        return x_j if edge_weight is None else tlx.reshape(edge_weight,(-1,1)) * x_j
        
