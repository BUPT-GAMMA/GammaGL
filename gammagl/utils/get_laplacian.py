# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/05/15 15:03
# @Author  : clear
# @FileName: get_laplacian.py
import tensorlayerx as tlx
from gammagl.mpops import *
from gammagl.utils.loop import add_self_loops

def get_laplacian(edge_index, num_nodes, edge_weight=None, normalization=None):
    """
        calculate GCN Normalization.

        Parameters
        ----------
        edge_index:
            edge index.
        num_nodes:
            number of nodes of graph.
        edge_weight:
            edge weights of graph.
        normalization:
            norm type: None, sym, rw.

        Returns
        --------
        edge_index: tensor
            the new edge index.
        edge_weight: tensor
            1-dim Tensor.

        """
    if normalization is not None:
        assert normalization in ['sym', 'rw']  # 'Invalid normalization'

    if edge_weight is None:
        edge_weight = tlx.ones(shape=(edge_index.shape[1], 1))  # torch backend `shape` should be tuple.


    row, col = edge_index[0], edge_index[1]
    deg = tlx.reshape(unsorted_segment_sum(edge_weight, row, num_segments=num_nodes), (-1,))


    if normalization is None:
        # L = D - A.
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_weight = tlx.concat([-edge_weight, deg], axis=0)
    elif normalization == 'sym':
        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt = tlx.pow(deg, -0.5)
        # deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = tlx.gather(deg_inv_sqrt, row) * edge_weight * tlx.gather(deg_inv_sqrt, col)

        # L = I - A_norm.
        edge_weight = tlx.reshape(edge_weight, (-1, 1))
        edge_index, tmp = add_self_loops(edge_index, edge_attr=-edge_weight, num_nodes=num_nodes)
        assert tmp is not None
        edge_weight = tlx.reshape(tmp, (-1,))
    else:
        # Compute A_norm = -D^{-1} A.
        deg_inv = tlx.pow(deg, -1)
        # deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = tlx.gather(deg_inv,row) * edge_weight
        edge_weight = tlx.reshape(edge_weight, (-1, 1))

        # L = I - A_norm.
        edge_index, tmp = add_self_loops(edge_index, edge_attr=-edge_weight, num_nodes=num_nodes)
        assert tmp is not None
        edge_weight = tlx.reshape(tmp, (-1,))

    return edge_index, edge_weight

