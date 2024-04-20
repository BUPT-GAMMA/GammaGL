import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import gcn_norm
from gammagl.utils.Calg import cal_g_gradient
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
        
