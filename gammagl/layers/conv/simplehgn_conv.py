import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax
import numpy as np

class SimpleHGNConv(MessagePassing):
    def __init__(self,
                in_feats,               #输入维度
                out_feats,              #输出维度
                num_etypes,             #边类型数
                edge_feats,             #边的嵌入表示维度
                heads=1,                #GAT头个数
                concat=True,            #concat需要处理，源码是在隐藏层处理完后，再计算了一次平均
                negative_slope=0.2,      #leaky_relu的斜率
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

        self.fc_node = tlx.nn.Linear(out_feats * heads, b_init=None, W_init=tlx.initializers.XavierNormal(gain=1.414))
        self.fc_edge = tlx.nn.Linear(edge_feats * heads, b_init=None, W_init=tlx.initializers.XavierNormal(gain=1.414))

        self.attn_src = self._get_weights('attn_l', shape=(1, heads, out_feats), init=tlx.initializers.XavierNormal(gain=1.414))
        self.attn_dst = self._get_weights('attn_r', shape=(1, heads, out_feats), init=tlx.initializers.XavierNormal(gain=1.414))
        self.attn_edge = self._get_weights('attn_e', shape=(1, heads, edge_feats), init=tlx.initializers.XavierNormal(gain=1.414))

        self.feat_drop = tlx.nn.Dropout(feat_drop)
        self.attn_drop = tlx.nn.Dropout(attn_drop)
        self.leaky_relu = tlx.nn.LeakyReLU(negative_slope)
        if residual:
            self.fc_res = tlx.nn.Linear(heads * out_feats, b_init=None, W_init=tlx.initializers.XavierNormal(gain=1.414)) if self.in_feats != self.out_feats else None#TODO:应该是identity()
        '''if(self.in_feats != out_feats):
            self.fc_res = tlx.nn.Linear(num_heads * out_feats, b_init=None, W_init=tlx.initializers.XavierNormal())
        else:
            #Note:这里有问题，fc_rec应该是一个层，可以让
            self.fc_res = tlx.convert_to_tensor(np.identity(in_feats))'''
        
        self.activation = activation
        
        if bias:
            self.bias = self._get_weights("bias", (1, heads, out_feats))
        self.beta = beta

    def message(self, x, edge_index, edge_x, res_alpha=None):
        node_src = edge_index[0, :]
        node_dst = edge_index[1, :]

        x = self.fc_node(x)
        x_src = tlx.ops.gather(x, node_src)
        x_dst = tlx.ops.gather(x, node_dst)
        x_src = self.feat_drop(x_src)
        x_dst = self.feat_drop(x_dst)
        edge_x = self.edge_embedding(edge_x)
        edge_x = self.fc_edge(edge_x).reshape(-1, self.heads, self.edge_feats)

        #计算权重alpha
        weight_src = tlx.reduce_sum(x_src * self.attn_src, -1)
        weight_dst = tlx.reduce_sum(x_dst * self.attn_dst, -1)
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
        
        if self.activation is not None:
            rst = self.activation(rst)
        return rst, alpha


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
        msg, alpha = self.message(**msg_kwargs)
        x = self.aggregate(msg, edge_index, num_nodes=kwargs['num_nodes'], aggr=aggr)
        x = self.update(x)
        return x, alpha

    def forward(self, x, edge_index, edge_feat, res_attn=None):
        return self.propagate(x, edge_index)
        
        

        


        
        



        
