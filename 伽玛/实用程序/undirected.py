from typing import Optional, Union, Tuple, List
import numpy as np
#import torch
#from torch import Tensor
#from torch_geometric.utils.coalesce import coalesce
import tensorlayerx as tlx
from gammagl.utils.coalesce import coalesce
from .num_nodes import maybe_num_nodes


def to_undirected(edge_index,edge_attr=None,num_nodes=None,reduce: str = "add"):
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will remove duplicates for all its entries.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (string, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :obj:`Tensor` or :obj:`List[Tensor]]`)
    """
    # Maintain backward compatibility to `to_undirected(edge_index, num_nodes)`
    if isinstance(edge_attr, int):
        edge_attr = None
        num_nodes = edge_attr

    row, col = edge_index
    row, col = np.concatenate([row, col], axis=0), np.concatenate([col, row], axis=0)
    edge_index = tlx.stack([row, col], axis=0)

    if edge_attr is not None and tlx.ops.is_tensor(edge_attr):
        edge_attr = np.concatenate([edge_attr, edge_attr], axis=0)
    elif edge_attr is not None:
        edge_attr = [np.concatenate([e, e], axis=0) for e in edge_attr]

    return coalesce(edge_index, edge_attr, num_nodes, reduce)
