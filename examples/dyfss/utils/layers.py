import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn.initializers import XavierUniform
from tensorlayerx.nn import Dropout


class GraphConvolution(Module):
    """
    多后端支持的 GCN Layer (TensorLayerX 版本)
    """

    def __init__(self, in_features, out_features, dropout=0.0, act=tlx.nn.ReLU()):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = Dropout(p=dropout)
        self.act = act

        # 初始化权重矩阵
        self.weight = self._get_weights("weight", shape=(in_features, out_features), init=XavierUniform())

    def forward(self, inputs, adj):
        x = self.dropout(inputs)
        support = tlx.matmul(x, self.weight)

        # 如果 adj 是稀疏的 scipy 矩阵或 COO 表达，需手动判断
        if hasattr(adj, 'to_dense'):
            adj = adj.to_dense()
        elif hasattr(adj, 'todense'):
            import numpy as np
            adj = tlx.convert_to_tensor(np.array(adj.todense(), dtype='float32'))

        output = tlx.matmul(adj, support)
        return self.act(output)

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features})"
