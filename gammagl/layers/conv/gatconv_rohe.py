import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax

class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads, dropout_rate=0., negative_slope=0.2, residual=False, settings=None):
        super(GATConv, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.settings = settings or {'K': 10, 'P': 0.6, 'tau': 0.1, 'Flag': "None", 'T': 1, 'TransM': None}
        self.feat_drop = tlx.layers.Dropout(dropout_rate)
        self.attn_drop = tlx.layers.Dropout(dropout_rate)
        self.leaky_relu = tlx.layers.LeakyReLU(negative_slope)

        # Linear transformation for input features
        self.fc = tlx.layers.Linear(in_features=in_channels, out_features=out_channels * num_heads, b_init=None)

        # Attention weights
        init = tlx.initializers.TruncatedNormal()
        self.attn_l = self._get_weights("attn_l", shape=(1, num_heads, out_channels), init=init)
        self.attn_r = self._get_weights("attn_r", shape=(1, num_heads, out_channels), init=init)

        if residual:
            self.res_fc = tlx.layers.Linear(in_features=in_channels, out_features=num_heads * out_channels, b_init=None)
        else:
            self.res_fc = None

    def mask(self, attM):
        T = self.settings['T']
        indices_to_remove = attM < tlx.topk(attM, T, sorted=False)[0][..., -1, None]
        attM = tlx.where(indices_to_remove, tlx.constant(-9e15, dtype=tlx.float32, shape=attM.shape), attM)
        return attM

    def forward(self, x, edge_index, num_nodes):
        edge_trans = self.settings['TransM']  # 从 settings 中提取转移矩阵
        edge_index = tlx.convert_to_tensor(edge_index, dtype=tlx.int64)
        # Apply linear transformation to node features
        x = self.fc(x)  # (N, num_heads * out_channels)
        x = tlx.reshape(x, (-1, self.num_heads, self.out_channels))  # (N, num_heads, out_channels)

        num_edges = edge_index.shape[1]
        
        node_src = edge_index[0, :]
        node_dst = edge_index[1, :]
    
        feat_src = tlx.gather(x, node_src)
        feat_dst = tlx.gather(x, node_dst)

        # Feature-based similarity
        e = tlx.reduce_sum(feat_src * feat_dst, axis=-1)  # (num_edges, num_heads)

        #  edge_trans 是一个稀疏矩阵的实例 (例如来自 scipy 的 csr_matrix)
        # 首先将其数据部分提取出来并转换为 Tensor
        edge_trans_data = tlx.convert_to_tensor(edge_trans.data, dtype=tlx.float32)  # 转换为 Tensor

        # 然后进行 reshape 操作，调整张量形状为 (num_edges, 1)
        edge_trans = tlx.reshape(edge_trans_data, (num_edges, 1))  # (num_edges, 1)

        # Apply the transition prior matrix
        edge_trans = edge_trans.repeat(1, self.num_heads)
    
        attn = e * edge_trans
        attn = self.mask(attn)
      
        alpha = segment_softmax(attn, node_dst, num_nodes)
        # Apply attention to node features
        out = self.propagate(x, edge_index, num_nodes=num_nodes, edge_weight=alpha)

        # 拼接所有 head 的输出，返回 (N, num_heads * out_channels)
        out = tlx.reshape(out, (-1, self.num_heads * self.out_channels))  # (N, num_heads * out_channels)

        if self.res_fc:
            res = self.res_fc(feat_dst)
            out += res

        return out
