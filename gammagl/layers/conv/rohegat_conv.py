# -*- coding: UTF-8 -*-
import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax
from gammagl.ops.functional import _unique_impl

class RoheGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads, dropout_rate=0., negative_slope=0.2, residual=False, settings=None):
        super(RoheGATConv, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.settings = settings or {'K': 10, 'P': 0.6, 'tau': 0.1, 'Flag': "None", 'T': 1, 'TransM': None}
        self.feat_drop = tlx.layers.Dropout(dropout_rate)
        self.attn_drop = tlx.layers.Dropout(dropout_rate)
        self.leaky_relu = tlx.layers.LeakyReLU(negative_slope)

        # 节点特征的线性变换
        self.fc = tlx.layers.Linear(in_features=in_channels, out_features=out_channels * num_heads, b_init=None)

        # 注意力权重
        init = tlx.initializers.TruncatedNormal()
        self.attn_l = self._get_weights("attn_l", shape=(num_heads, out_channels), init=init)
        self.attn_r = self._get_weights("attn_r", shape=(num_heads, out_channels), init=init)

        if residual:
            self.res_fc = tlx.layers.Linear(in_features=in_channels, out_features=num_heads * out_channels, b_init=None)
        else:
            self.res_fc = None

    def forward(self, x, edge_index, num_nodes):
        edge_trans_values = self.settings['TransM']  # 这是与 edge_index 对应的 (num_edges,) 的张量
        T = self.settings['T']  # 每个节点保留的边数量
        print(edge_trans_values)
        # 对节点特征应用线性变换
        x = self.fc(x)  # (N, num_heads * out_channels)
        x = tlx.reshape(x, (-1, self.num_heads, self.out_channels))  # (N, num_heads, out_channels)
        edge_index = tlx.convert_to_tensor(edge_index, dtype=tlx.int64)

        # 使用边索引提取特征
        node_src = edge_index[0, :]
        node_dst = edge_index[1, :]
        feat_src = tlx.gather(x, node_src)  # 提取源节点的特征 (num_edges, num_heads, out_channels)
        feat_dst = tlx.gather(x, node_dst)  # 提取目标节点的特征 (num_edges, num_heads, out_channels)

        # 基于特征相似性计算 e（使用内积代替简单加和）
        e = tlx.reduce_sum(feat_src * feat_dst, axis=-1)  # (num_edges, num_heads)

        # 使用 LeakyReLU 非线性激活函数
        e = self.leaky_relu(e)  # (num_edges, num_heads)

        # 对注意力分数应用转移矩阵
        edge_trans_values = tlx.convert_to_tensor(edge_trans_values, dtype=tlx.float32)  # (num_edges,)
        edge_trans_values = tlx.reshape(edge_trans_values, (-1, 1))  # (num_edges, 1)
        edge_trans_values = edge_trans_values.repeat(1, self.num_heads)  # (num_edges, num_heads)
        attn = e * edge_trans_values  # (num_edges, num_heads)
       
        # 使用 _unique_impl 获取唯一节点并计算每个节点的边的个数
        node_dst_numpy = tlx.convert_to_numpy(node_dst)  # 将 tensor 转换为 numpy.ndarray
        unique_nodes, inverse, counts = _unique_impl(node_dst_numpy, sorted=False, return_inverse=True, return_counts=True)

        mask = tlx.zeros_like(attn)  # (num_edges, num_heads)
      
        for i, node in enumerate(unique_nodes):
            node_edges_idx = (inverse == i).nonzero().view(-1)
            for h in range(attn.shape[1]):  # for each head
                attn_head = attn[:, h]
                if len(node_edges_idx) <= T:
                    mask[node_edges_idx, h] = 1
                else:
                    topk_idx = tlx.argsort(attn_head[node_edges_idx], descending=True)[:T]
                    selected_idx = node_edges_idx[topk_idx]
                    mask[selected_idx, h] = 1

        # 应用掩码并将掩码的元素设置为非常小的值
        very_negative_value = -9e15
        attn = tlx.where(mask > 0, attn, very_negative_value)

        # 对目的节点应用 softmax 操作
        alpha = segment_softmax(attn, node_dst, num_nodes)
       
        # 将注意力应用于节点特征
        out = self.propagate(x, edge_index, num_nodes=num_nodes, edge_weight=alpha)

        # 重塑并应用残差连接（如果有）
        out = tlx.reshape(out, (-1, self.num_heads * self.out_channels))
        if self.res_fc:
            res = self.res_fc(x)
            out += res

        return out
