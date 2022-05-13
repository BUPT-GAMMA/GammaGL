from typing import Optional, Tuple, Union
import tensorlayerx as tlx
import numpy as np
from .num_nodes import maybe_num_nodes
def add_self_loops(
        edge_index, n_loops=None, edge_attr=None,
        fill_value: Union[float, str] = None,
        num_nodes: Optional[int] = None):
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted or has multi-dimensional edge features
    (:obj:`edge_attr != None`), edge features of self-loops will be added
    according to :obj:`fill_value`.
    
    .. code-block:: python

        >>> from gammagl.data import Graph
        >>> from gammagl.utils.loop import add_self_loops
        >>> import numpy
        >>> edge_index = tlx.constant([[0, 0, 0], [1, 2, 3]])
        array([[0, 0, 0],
                [1, 2, 3]])
        >>> edge_index, _ = add_self_loops(edge_index)
        array([[0, 0, 0, 0, 1, 2, 3],
               [1, 2, 3, 0, 1, 2, 3]])

        
    Args:
        edge_index (LongTensor): The edge indices.
        n_loops (int): the number of loops
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_attr != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    N = maybe_num_nodes(edge_index, num_nodes)

    # loop_index = tlx.convert_to_tensor(np.arange(0, N), dtype=tlx.int64)
    # edge_index = tlx.convert_to_tensor(edge_index, dtype=tlx.int64) # 否则，torch 可能报错
    loop_index = tlx.convert_to_tensor([np.arange(N).repeat(1),
                                             np.arange(N).repeat(1)], dtype=edge_index.dtype)

    if edge_attr is not None:
        if fill_value is None:
            loop_attr = tlx.ones((N, edge_attr.shape[1]), dtype=edge_attr.dtype)
        elif isinstance(fill_value, (int, float)):
            # loop_attr = edge_attr.new_full((N, ) + edge_attr.size()[1:],
            #                                fill_value)
            loop_attr = tlx.constant(fill_value, (N,) + edge_attr.shape[1], dtype=edge_attr.dtype)
        # elif isinstance(fill_value, Tensor):
        #     loop_attr = fill_value.to(edge_attr.device, edge_attr.dtype)
        #     if edge_attr.dim() != loop_attr.dim():
        #         loop_attr = loop_attr.unsqueeze(0)
        #     sizes = [N] + [1] * (loop_attr.dim() - 1)
        #     loop_attr = loop_attr.repeat(*sizes)

        # elif isinstance(fill_value, str):
        #     loop_attr = scatter(edge_attr, edge_index[1], dim=0, dim_size=N,
        #                         reduce=fill_value)
        else:
            raise AttributeError("No valid 'fill_value' provided")

        edge_attr = tlx.concat([edge_attr, loop_attr], axis=0)

    edge_index = tlx.concat([edge_index, loop_index], axis=1)
    return edge_index, edge_attr
