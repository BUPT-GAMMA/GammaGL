import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax
import numpy as np

class SimpleHGNConv(MessagePassing):
    def __init__(self,
                in_feats,
                out_feats,
                num_etypes,
                edge_feats,
                heads=1,
                concat=True,            #concat需要处理，源码是在隐藏层处理完后，再计算了一次平均
                negetive_slop=0.2,
                feat_drop=0.,
                attn_drop=0.,
                residual=False,
                activation=None,
                bias=None,
                beta=0.,):
        super().__init__()

        self.out_channels = out_feats
        self.num_etypes = num_etypes
        self.edge_feats = edge_feats
        self.heads = heads

        self.edge_embedding = tlx.nn.Embedding(num_etypes, edge_feats)
        #TODO:初始化参数需要的gain，tlx框架下没有提供
        self.fc_node = tlx.nn.Linear(out_feats * heads, b_init=None, W_init=tlx.initializers.XavierNormal(gain=1.414))
        self.fc_edge = tlx.nn.Linear(edge_feats * heads, b_init=None, W_init=tlx.initializers.XavierNormal(gain=1.414))

        self.attn_src = self._get_weights('attn_l', shape=(1, heads, out_feats), init=tlx.initializers.XavierNormal(gain=1.414))
        self.attn_dst = self._get_weights('attn_r', shape=(1, heads, out_feats), init=tlx.initializers.XavierNormal(gain=1.414))
        self.attn_edge = self._get_weights('attn_e', shape=(1, heads, edge_feats), init=tlx.initializers.XavierNormal(gain=1.414))

        self.feat_drop = tlx.nn.Dropout(feat_drop)
        self.attn_drop = tlx.nn.Dropout(attn_drop)
        self.leaky_relu = tlx.nn.LeakyReLU(negative_slope)

        
        if(self.in_feats != out_feats):
            self.fc_res = tlx.nn.Linear(num_heads * out_feats, b_init=None, W_init=tlx.initializers.XavierNormal())
        else:
            self.fc_res = tlx.convert_to_tensor(np.identity(in_feats))
        
        self.activation = activation
        
        if bias:
            self.bias = self._get_weights("bias", (1, heads, out_feats))
        self.beta = beta

    def message(self, x, edge_index, edge_feat, res_attn=None):
        node_src = edge_index[0, :]
        node_dst = edge_index[1, :]

        #dropout 输入
        x_src = tlx.ops.gather(x, node_src)
        x_dst = tlx.ops.gather(x, node_dst)
        x_src = self.feat_drop(x_src)
        x_dst = self.feat_drop(x_dst)
        edge_feat = self.edge_embedding(edge_feat)
        edge_feat = self.fc_edge(edge_feat).reshape(-1, self._num_heads, self._edge_feats)

        #计算权重alpha
        weight_src = tlx.reduce_sum(x_src * self.attn_src, -1)
        weight_dst = tlx.reduce_sum(x_dst * self.attn_src, -1)
        weight_edge = tlx.reduce_sum(edge_feat * self.attn_edge, -1)
        weight = self.leaky_relu(weight_src + weight_dst + weight_edge)
        alpha = self.dropout(segment_softmax(weight, node_dst, num_nodes))

        #注意力残差
        if res_attn is not None:
            alpha = alpha * (1 - self.beta) + res_attn * self.beta

        #节点级别残差
        rst = x_src * alpha
        if self.fc_res is not None:
            res_val = self.fc_res(x_dst)
            rst  = rst + res_val

        if self.bias:
            rst = rst + self.bias
        
        if self.activation:
            rst = self.activation(rst)

        return rst, alpha


    def forward(self, x, edge_index, edge_feat, res_attn=None):
        TODO()
        
        

        


        
        



        
