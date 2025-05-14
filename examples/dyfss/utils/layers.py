import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn.initializers import XavierUniform
from tensorlayerx.nn import Dropout


class GraphConvolution(Module):


    def __init__(self, in_features, out_features, dropout=0.0, act=tlx.nn.ReLU()):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = Dropout(p=dropout)
        self.act = act
        self.weight = self._get_weights("weight", shape=(in_features, out_features), init=XavierUniform())

    def forward(self, inputs, adj):
        x = self.dropout(inputs)
        support = tlx.matmul(x, self.weight)

        if hasattr(adj, 'to_dense'):
            adj = adj.to_dense()
        elif hasattr(adj, 'todense'):
            import numpy as np
            adj = tlx.convert_to_tensor(np.array(adj.todense(), dtype='float32'))

        output = tlx.matmul(adj, support)
        return self.act(output)

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features})"
