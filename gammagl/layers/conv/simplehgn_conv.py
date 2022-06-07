import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax
import numpy as np
from gammagl.mpops import *

class SimpleHGNConv(MessagePassing):
    def __init__(self,
                in_feats,               #输入维度
                out_feats,              #输出维度
                num_etypes,             #边类型数
                edge_feats,             #边的嵌入表示维度
                heads=1,                #GAT头个数
                concat=True,            #concat需要处理，源码是在隐藏层处理完后，再计算了一次平均
                negative_slope=0.2,     #leaky_relu的斜率
                feat_drop=0.,           #节点特征的drop_rate
                attn_drop=0.,           #attention的drop_rate
                residual=False,         #是否残差
                activation=None,        #激活函数
                bias=None,              #是否偏置
                beta=0.,):              #超参数，beta
        super().__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.edge_feats = edge_feats
        self.heads = heads
        self.out_feats = out_feats
        self.edge_embedding = tlx.nn.Embedding(num_etypes, edge_feats)

        #Note:需要指明in_features，否则不会被放入trainable_weights
        self.fc_node = tlx.nn.Linear(out_feats * heads, in_features=in_feats, b_init=None, W_init=tlx.initializers.XavierNormal(gain=1.414), name='fc_node')
        self.fc_edge = tlx.nn.Linear(edge_feats * heads, in_features=edge_feats, b_init=None, W_init=tlx.initializers.XavierNormal(gain=1.414), name='fc_edge')

        self.attn_src = self._get_weights('attn_l', shape=(1, heads, out_feats), init=tlx.initializers.XavierNormal(gain=1.414), order=True)
        self.attn_dst = self._get_weights('attn_r', shape=(1, heads, out_feats), init=tlx.initializers.XavierNormal(gain=1.414), order=True)
        self.attn_edge = self._get_weights('attn_e', shape=(1, heads, edge_feats), init=tlx.initializers.XavierNormal(gain=1.414), order=True)

        self.feat_drop = tlx.nn.Dropout(feat_drop)
        self.attn_drop = tlx.nn.Dropout(attn_drop)
        self.leaky_relu = tlx.nn.LeakyReLU(negative_slope)
        #if residual:
        self.fc_res = tlx.nn.Linear(heads * out_feats, in_features=in_feats, b_init=None, W_init=tlx.initializers.XavierNormal(gain=1.414), name='fc_res') if self.in_feats != self.out_feats else None#TODO:应该是identity()
        
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

        #计算权重alpha
        node_src = edge_index[0, :]
        node_dst = edge_index[1, :]
        weight_src = tlx.ops.gather(tlx.reduce_sum(x_new * self.attn_src, -1), node_src)
        weight_dst = tlx.ops.gather(tlx.reduce_sum(x_new * self.attn_dst, -1), node_dst)
        weight_edge = tlx.reduce_sum(edge_feat * self.attn_edge, -1)
        weight = self.leaky_relu(weight_src + weight_dst + weight_edge)
        alpha = self.attn_drop(segment_softmax(weight, node_dst, num_nodes))

        #注意力残差
        if res_alpha is not None:
            alpha = alpha * (1 - self.beta) + res_attn * self.beta

        #节点级别残差
        rst = tlx.ops.gather(x_new, node_src) * tlx.ops.expand_dims(alpha, axis=-1)
        rst = unsorted_segment_sum(rst, node_dst, num_nodes)
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
        #x_new = self.fc_node(x)
        return self.propagate(x, edge_index, edge_feat=edge_feat)
        
        
