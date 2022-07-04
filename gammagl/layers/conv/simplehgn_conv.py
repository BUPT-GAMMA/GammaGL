import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax
from gammagl.mpops import *


class SimpleHGNConv(MessagePassing):
    r'''The SimpleHGN layer from the `"Are we really making much progress? Revisiting, benchmarking, and refining heterogeneous graph neural networks"
    <https://dl.acm.org/doi/pdf/10.1145/3447548.3467350>`_ paper

    The model extend the original graph attention mechanism in GAT by including edge type information into attention calculation.

    Calculating the coefficient:
        
    .. math::
        \alpha_{ij} = \frac{exp(LeakyReLU(a^T[Wh_i||Wh_j||W_r r_{\psi(<i,j>)}]))}{\Sigma_{k\in\mathcal{E}}{exp(LeakyReLU(a^T[Wh_i||Wh_k||W_r r_{\psi(<i,k>)}]))}}  (1)
    
    Residual connection including Node residual:
    
    .. math::
        h_i^{(l)} = \sigma(\Sigma_{j\in \mathcal{N}_i} {\alpha_{ij}^{(l)}W^{(l)}h_j^{(l-1)}} + h_i^{(l-1)})  (2)
    
    and Edge residual:
        
    .. math::
        \alpha_{ij}^{(l)} = (1-\beta)\alpha_{ij}^{(l)}+\beta\alpha_{ij}^{(l-1)}  (3)
        
    Multi-heads:
    
    .. math::
        h^{(l+1)}_j = \parallel^M_{m = 1}h^{(l + 1, m)}_j  (4)
    
    Residual:
    
    .. math::
        h^{(l+1)}_j = h^{(l)}_j + \parallel^M_{m = 1}h^{(l + 1, m)}_j  (5)

    Parameters
    ----------
    in_feats: int
        the input dimension
    out_feats: int
        the output dimension
    num_etypes: int
        the number of the edge type
    edge_feats: int
        the edge dimension
    heads: int
        the number of heads in this layer
    negative_slope: float
        the negative slope used in the LeakyReLU
    feat_drop: float
        the feature drop rate
    attn_drop: float
        the attention score drop rate
    residual: boolean
        whether we need the residual operation
    activation:
        the activation function
    bias:
        whether we need the bias
    beta: float
        the hyperparameter used in edge residual
    
    '''
    def __init__(self,
                in_feats,
                out_feats,
                num_etypes,
                edge_feats,
                heads=1,
                negative_slope=0.2,
                feat_drop=0.,
                attn_drop=0.,
                residual=False,
                activation=None,
                bias=False,
                beta=0.,):
        super().__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.edge_feats = edge_feats
        self.heads = heads
        self.out_feats = out_feats
        self.edge_embedding = tlx.nn.Embedding(num_etypes, edge_feats)

        self.fc_node = tlx.nn.Linear(out_feats * heads, in_features=in_feats, b_init=None, W_init=tlx.initializers.XavierNormal(gain=1.414))
        self.fc_edge = tlx.nn.Linear(edge_feats * heads, in_features=edge_feats, b_init=None, W_init=tlx.initializers.XavierNormal(gain=1.414))

        self.attn_src = self._get_weights('attn_l', shape=(1, heads, out_feats), init=tlx.initializers.XavierNormal(gain=1.414), order=True)
        self.attn_dst = self._get_weights('attn_r', shape=(1, heads, out_feats), init=tlx.initializers.XavierNormal(gain=1.414), order=True)
        self.attn_edge = self._get_weights('attn_e', shape=(1, heads, edge_feats), init=tlx.initializers.XavierNormal(gain=1.414), order=True)

        self.feat_drop = tlx.nn.Dropout(feat_drop)
        self.attn_drop = tlx.nn.Dropout(attn_drop)
        self.leaky_relu = tlx.nn.LeakyReLU(negative_slope)

        self.fc_res = tlx.nn.Linear(heads * out_feats, in_features=in_feats, b_init=None, W_init=tlx.initializers.XavierNormal(gain=1.414)) if residual else None
        
        self.activation = activation
        
        self.bias = self._get_weights("bias", (1, heads, out_feats)) if bias else None
        self.beta = beta

    def message(self, x, edge_index, edge_feat, num_nodes, res_alpha=None):
        x_new = self.fc_node(x)
        x_new = tlx.ops.reshape(x_new, shape=[-1, self.heads, self.out_feats])
 
        x_new = self.feat_drop(x_new)
        edge_feat = self.edge_embedding(edge_feat)
        edge_feat = self.fc_edge(edge_feat)
        edge_feat = tlx.ops.reshape(edge_feat, [-1, self.heads, self.edge_feats])

        #calculate the alpha
        node_src = edge_index[0, :]
        node_dst = edge_index[1, :]
        weight_src = tlx.ops.gather(tlx.reduce_sum(x_new * self.attn_src, -1), node_src)
        weight_dst = tlx.ops.gather(tlx.reduce_sum(x_new * self.attn_dst, -1), node_dst)
        weight_edge = tlx.reduce_sum(edge_feat * self.attn_edge, -1)
        weight = self.leaky_relu(weight_src + weight_dst + weight_edge)
        alpha = self.attn_drop(segment_softmax(weight, node_dst, num_nodes))

        #edge residual
        if res_alpha is not None:
            alpha = alpha * (1 - self.beta) + res_alpha * self.beta

        rst = tlx.ops.gather(x_new, node_src) * tlx.ops.expand_dims(alpha, axis=-1)
        rst = unsorted_segment_sum(rst, node_dst, num_nodes)
        #node residual
        if self.fc_res is not None:
            res_val = self.fc_res(x)
            res_val = tlx.ops.reshape(res_val, shape=[x.shape[0], -1, self.out_feats])
            rst  = rst + res_val

        if self.bias is not None:
            rst = rst + self.bias
        
        if self.activation is not None:
            rst = self.activation(rst)
        x = rst
        return x, alpha


    def propagate(self, x, edge_index, aggr='sum', **kwargs):
        """
        Function that perform message passing. 
        Args:
            x: input node feature
            edge_index: edges from src to dst
            aggr: aggregation type, default='sum', optional=['sum', 'mean', 'max']
            kwargs: other parameters dict

        """

        if 'num_nodes' not in kwargs.keys() or kwargs['num_nodes'] is None:
            kwargs['num_nodes'] = x.shape[0]

        coll_dict = self.__collect__(x, edge_index, aggr, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        x, alpha = self.message(**msg_kwargs)
        x = self.update(x)
        return x, alpha

    def forward(self, x, edge_index, edge_feat, res_attn=None):
        return self.propagate(x, edge_index, edge_feat=edge_feat)
        
        
