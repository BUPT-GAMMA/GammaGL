import os
os.environ.setdefault('TL_BACKEND', 'torch')

import numpy as np
import tensorlayerx as tlx
import math
import torch
from gammagl.layers.manifolds import Lorentz, Sphere
from gammagl.mpops import unsorted_segment_sum
from gammagl.utils.softmax import segment_softmax


class EuclideanEncoder(tlx.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bias=True, activation=tlx.relu, dropout=0.1):
        super().__init__()
        self.lin = tlx.layers.Linear(in_features=in_dim, out_features=hidden_dim, b_init=tlx.initializers.constant(1.0) if bias else None)
        self.activation = activation
        self.proj = tlx.layers.Linear(in_features=hidden_dim, out_features=out_dim, b_init=tlx.initializers.constant(1.0) if bias else None)
        self.dropout = tlx.nn.Dropout(p=dropout)
        self.drop = dropout

    def forward(self, x):
        x = self.lin(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.proj(self.dropout(x))
        x = x / (tlx.sqrt(tlx.reduce_sum(x * x, axis=-1, keepdims=True)) + 1e-8)
        return x


class ManifoldEncoder(tlx.nn.Module):
    def __init__(self, manifold, in_dim, hidden_dim, out_dim, bias=True, activation=None, dropout=0.1):
        super().__init__()
        self.manifold = manifold
        self.lin = ConstCurveLinear(manifold, in_dim, out_dim, bias=bias, dropout=dropout, activation=activation)
        self.agg = ConstCurveAgg(manifold, out_dim, dropout, use_att=False)

    def forward(self, x, edge_index):
        x = self.manifold.expmap0(x)
        x = self.lin(x)
        x = self.agg(x, edge_index)
        return x


class ConstCurveLinear(tlx.nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.0,
                 scale=10,
                 fixscale=False,
                 activation=None):
        super().__init__()
        self.manifold = manifold
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = tlx.layers.Linear(
            in_features=self.in_features, out_features=self.out_features, b_init=tlx.initializers.constant(1.0) if bias else None)
        self.reset_parameters()
        self.dropout = tlx.nn.Dropout(p=dropout)
        self.scale = tlx.nn.Parameter(tlx.ones((1,)) * math.log(scale))
        self.sign = -1. if isinstance(manifold, Lorentz) else 1.

    def forward(self, x):
        if self.activation is not None:
            x = self.activation(x)
        x = self.dropout(x)
        x = self.weight(x)
        x_narrow = x[..., 1:]
        time = tlx.sigmoid(x[..., :1]) * tlx.exp(self.scale) + 1.1 if isinstance(self.manifold, Lorentz)  \
        else tlx.sigmoid(x[..., :1]) - 0.5
        scale = self.sign * (1. / self.manifold.k - time * time) / \
            tlx.maximum(tlx.reduce_sum(x_narrow * x_narrow, axis=-1, keepdims=True), tlx.convert_to_tensor(1e-8))
        x = tlx.concat([time, x_narrow * tlx.sqrt(scale)], axis=-1)
        return x

    def reset_parameters(self):
        pass


class ConstCurveAgg(tlx.nn.Module):
    def __init__(self, manifold, in_features, dropout=0.0, use_att=False):
        super(ConstCurveAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_features
        self.dropout = dropout
        self.use_att = use_att
        if self.use_att:
            self.key_linear = ConstCurveLinear(manifold, in_features, in_features)
            self.query_linear = ConstCurveLinear(manifold, in_features, in_features)
            self.bias = tlx.nn.Parameter(tlx.zeros((1,)) + 20)
            self.scale = tlx.nn.Parameter(tlx.zeros((1,)) + math.sqrt(in_features))
        if isinstance(manifold, Lorentz):
            self.neg_dist = lambda x, y: 2 + 2 * manifold.cinner(x, y)
        else:
            self.neg_dist = lambda x, y: -manifold.dist(x, y) ** 2
        self.sign = -1. if isinstance(manifold, Lorentz) else 1.

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        if isinstance(src, torch.Tensor):
            src = src.to(dtype=torch.int64)
            dst = dst.to(dtype=torch.int64)
        else:
            if hasattr(src, 'detach'):
                src = src.detach().cpu().numpy()
                dst = dst.detach().cpu().numpy()
            elif hasattr(src, 'numpy'):
                src = src.numpy()
                dst = dst.numpy()
            src = tlx.convert_to_tensor(src, dtype=tlx.int64)
            dst = tlx.convert_to_tensor(dst, dtype=tlx.int64)
        if self.use_att:
            query = self.query_linear(x)
            key = self.key_linear(x)
            att_adj = 2 + 2 * self.manifold.cinner(tlx.gather(query, dst), tlx.gather(key, src))
            att_adj = att_adj / self.scale + self.bias
            att_adj = tlx.sigmoid(att_adj)
            support_t = unsorted_segment_sum(att_adj * tlx.gather(x, dst), src, x.shape[0])
        else:
            support_t = unsorted_segment_sum(tlx.gather(x, dst), src, x.shape[0])

        denorm = self.sign * self.manifold.inner(None, support_t, keepdim=True)

        denorm = tlx.sqrt(tlx.maximum(tlx.abs(denorm), tlx.convert_to_tensor(1e-8)))
        output = 1. / tlx.sqrt(self.manifold.k) * support_t / denorm
        return output
