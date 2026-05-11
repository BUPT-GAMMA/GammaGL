import os
os.environ.setdefault('TL_BACKEND', 'torch')

import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.layers.conv.rgt_layers import ConstCurveLinear
from gammagl.mpops import unsorted_segment_sum
from gammagl.utils.softmax import segment_softmax


def segment_softmax_fast(data, segment_ids, num_segments=None, eps=1e-12):
    """Segment softmax with compact segment ids."""
    if num_segments is None:
        num_segments = int(tlx.convert_to_numpy(tlx.reduce_max(segment_ids))) + 1
    return segment_softmax(data, segment_ids, num_segments=num_segments)


class HyperbolicStructureLearner(nn.Module):
    def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(HyperbolicStructureLearner, self).__init__()
        self.manifold_H = manifold_H
        self.manifold_S = manifold_S
        self.tree_agg = CrossManifoldAttention(manifold_S, manifold_H, in_dim, hidden_dim, out_dim, dropout)

    def forward(self, x_H, x_S, batch_tree):
        """
        Local Attention based on BFS tree structure inherit from a sub-graph.
        :param x_H: Hyperbolic representation of nodes
        :param batch_tree: a batch graph with tree-graphs from one graph.
        :return: New Hyperbolic representation of nodes.
        """
        num_seeds = len(batch_tree)
        node_labels = tlx.tile(
            tlx.arange(start=0, limit=int(x_H.shape[0]), delta=1, dtype=tlx.int64),
            [num_seeds]
        )
        x = tlx.gather(x_H, node_labels)
        att_index = batch_tree.edge_index
        if hasattr(att_index, 'numpy'):
            att_index = att_index.numpy()
        x = self.tree_agg(tlx.gather(x_S, node_labels), x, x, edge_index=att_index)

        x_extend = tlx.concat([x, x_H], axis=0)
        label_extend = tlx.concat(
            [node_labels, tlx.arange(start=0, limit=int(x_H.shape[0]), delta=1, dtype=tlx.int64)],
            axis=0)
        z_H = self.manifold_H.Frechet_mean(x_extend, keepdim=True, sum_idx=label_extend)
        return z_H


class SphericalStructureLearner(nn.Module):
    """
    in_dim = out_dim
    """
    def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(SphericalStructureLearner, self).__init__()
        self.manifold_H = manifold_H
        self.manifold_S = manifold_S
        self.cycle_agg = CrossManifoldAttention(manifold_H, manifold_S, in_dim, hidden_dim, out_dim, dropout)

    def forward(self, x_H, x_S, batch_cycle):
        """
        :param x_H: Hyperbolic representation of nodes
        :param x_S: Sphere representation of nodes
        :param batch_cycle: a batch graph with cycles from one graph.
        :return: New sphere representation of nodes.
        """
        num_seeds = len(batch_cycle)
        node_labels = tlx.tile(
            tlx.arange(start=0, limit=int(x_S.shape[0]), delta=1, dtype=tlx.int64),
            [num_seeds]
        )
        x = tlx.gather(x_S, node_labels)
        att_index = batch_cycle.edge_index
        if hasattr(att_index, 'numpy'):
            att_index = att_index.numpy()
        x = self.cycle_agg(tlx.gather(x_H, node_labels), x, x, edge_index=att_index)

        x_extend = tlx.concat([x, x_S], axis=0)
        label_extend = tlx.concat(
            [node_labels, tlx.arange(start=0, limit=int(x_S.shape[0]), delta=1, dtype=tlx.int64)],
            axis=0)
        z_S = self.manifold_S.Frechet_mean(x_extend, keepdim=True, sum_idx=label_extend)

        return z_S


class EuclideanStructureLearner(nn.Module):
    def __init__(self, manifold_E, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(EuclideanStructureLearner, self).__init__()
        self.manifold_E = manifold_E
        self.sequence_agg = EuclideanAttention(manifold_E, in_dim, hidden_dim, out_dim, dropout)

    def forward(self, x_E, batch_sequence):
        """
        Local Attention based on BFS sequence structure inherit from a sub-graph.
        :param x_E: Euclidean representation of nodes
        :param batch_sequence: a batch graph with sequence from one graph.
        :return: New Euclidean representation of nodes.
        """
        num_seeds = len(batch_sequence)
        node_labels = tlx.tile(
            tlx.arange(start=0, limit=int(x_E.shape[0]), delta=1, dtype=tlx.int64),
            [num_seeds]
        )
        x = tlx.gather(x_E, node_labels)
        att_index = batch_sequence.edge_index
        if hasattr(att_index, 'numpy'):
            att_index = att_index.numpy()
        x = self.sequence_agg(x, x, x, edge_index=att_index)

        x_extend = tlx.concat([x, x_E], axis=0)
        label_extend = tlx.concat(
            [node_labels, tlx.arange(start=0, limit=int(x_E.shape[0]), delta=1, dtype=tlx.int64)],
            axis=0)
        z_E = self.manifold_E.Frechet_mean(x_extend, keepdim=True, sum_idx=label_extend)
        return z_E


class CrossManifoldAttention(nn.Module):
    def __init__(self, manifold_q, manifold_k, in_dim, hidden_dim, out_dim, dropout):
        super(CrossManifoldAttention, self).__init__()
        self.manifold_q = manifold_q
        self.manifold_k = manifold_k
        self.q_lin = ConstCurveLinear(manifold_q, in_dim, hidden_dim, bias=False, dropout=dropout)
        self.k_lin = ConstCurveLinear(manifold_k, in_dim, hidden_dim, bias=False, dropout=dropout)
        self.v_lin = ConstCurveLinear(manifold_k, in_dim, hidden_dim, bias=False, dropout=dropout)
        self.scalar_map = nn.Sequential(
            tlx.layers.Linear(in_features=2 * hidden_dim, out_features=1, b_init=None),
            tlx.nn.LeakyReLU(negative_slope=0.2)
        )
        self.proj = ConstCurveLinear(manifold_k, hidden_dim, out_dim, bias=False, dropout=dropout)

    def forward(self, x_q, x_k, x_v, edge_index, agg_index=None):
        q = self.q_lin(x_q)
        k = self.k_lin(x_k)
        v = self.v_lin(x_v)
        src, dst = edge_index[0], edge_index[1]
        src = tlx.cast(tlx.convert_to_tensor(src), tlx.int64)
        dst = tlx.cast(tlx.convert_to_tensor(dst), tlx.int64)
        agg_index = agg_index if agg_index is not None else src

        if int(src.shape[0]) == 0:
            out = tlx.zeros((q.shape[0], v.shape[-1]), dtype=v.dtype)
            denorm = self.manifold_k.inner(None, out, keepdim=True)
            denorm = tlx.sqrt(tlx.maximum(tlx.abs(denorm), tlx.convert_to_tensor(1e-8, dtype=denorm.dtype)))
            out = 1. / tlx.sqrt(self.manifold_k.k) * out / denorm
            return self.proj(out)

        qk = tlx.concat([tlx.gather(q, src), tlx.gather(k, dst)], axis=-1)
        score = tlx.squeeze(self.scalar_map(qk), axis=-1)
        src_np = tlx.convert_to_numpy(src)
        src_compact = tlx.convert_to_tensor(np.unique(src_np, return_inverse=True)[1], dtype=tlx.int64)
        num_segments = int(tlx.convert_to_numpy(tlx.reduce_max(src_compact))) + 1
        score = segment_softmax_fast(score, src_compact, num_segments=num_segments)

        out = unsorted_segment_sum(score[:, None] * tlx.gather(v, dst), agg_index, q.shape[0])

        denorm = self.manifold_k.inner(None, out, keepdim=True)
        denorm = tlx.sqrt(tlx.maximum(tlx.abs(denorm), tlx.convert_to_tensor(1e-8)))
        out = 1. / tlx.sqrt(self.manifold_k.k) * out / denorm
        out = self.proj(out)
        return out


class EuclideanAttention(nn.Module):
    def __init__(self, manifold_E, in_dim, hidden_dim, out_dim, dropout):
        super(EuclideanAttention, self).__init__()
        self.q_lin = tlx.layers.Linear(in_features=in_dim, out_features=hidden_dim, b_init=None)
        self.k_lin = tlx.layers.Linear(in_features=in_dim, out_features=hidden_dim, b_init=None)
        self.v_lin = tlx.layers.Linear(in_features=in_dim, out_features=hidden_dim, b_init=None)
        self.manifold_E = manifold_E
        self.scalar_map = nn.Sequential(
            tlx.layers.Linear(in_features=2 * hidden_dim, out_features=1, b_init=None),
            tlx.nn.LeakyReLU(negative_slope=0.2)
        )
        self.proj = tlx.layers.Linear(in_features=hidden_dim, out_features=out_dim, b_init=None)
        self.dropout = tlx.nn.Dropout(p=dropout)

    def forward(self, x_q, x_k, x_v, edge_index, agg_index=None):
        q = self.q_lin(x_q)
        k = self.k_lin(x_k)
        v = self.v_lin(x_v)
        src, dst = edge_index[0], edge_index[1]
        src = tlx.cast(tlx.convert_to_tensor(src), tlx.int64)
        dst = tlx.cast(tlx.convert_to_tensor(dst), tlx.int64)

        agg_index = agg_index if agg_index is not None else src
        if int(src.shape[0]) == 0:
            out = tlx.zeros((q.shape[0], v.shape[-1]), dtype=v.dtype)
            out = self.proj(out)
            out = self.dropout(out)
            out = out / (tlx.sqrt(tlx.reduce_sum(out * out, axis=-1, keepdims=True)) + 1e-8)
            return out

        qk = tlx.concat([tlx.gather(q, src), tlx.gather(k, dst)], axis=-1)
        score = tlx.squeeze(self.scalar_map(qk), axis=-1)
        src_np = tlx.convert_to_numpy(src)
        src_compact = tlx.convert_to_tensor(np.unique(src_np, return_inverse=True)[1], dtype=tlx.int64)
        score = segment_softmax_fast(score, src_compact, num_segments=int(tlx.convert_to_numpy(tlx.reduce_max(src_compact))) + 1)

        out = unsorted_segment_sum(score[:, None] * tlx.gather(v, dst), agg_index, q.shape[0])
        out = self.proj(out)
        out = self.dropout(out)
        out = out / (tlx.sqrt(tlx.reduce_sum(out * out, axis=-1, keepdims=True))+1e-8)

        return out
