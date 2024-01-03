from typing import Optional, Tuple, Union
import tensorlayerx as tlx
import numpy as np

from gammagl.utils.check import check_is_numpy
from .num_nodes import maybe_num_nodes


def contains_self_loops(edge_index) -> bool:
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` contains
    self-loops.
    Args:
        edge_index (LongTensor): The edge indices.
    :rtype: bool
    """
    mask = edge_index[0] == edge_index[1]
    return tlx.any(mask, axis=0)


def remove_self_loops(edge_index, edge_attr=None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
    :rtype: edge_index(Tensor if edge_index inputted is Tensor
            || np.ndarray if edge_index inputted is np.ndarray)
    """
    mask = edge_index[0] != edge_index[1]
    if tlx.is_tensor(edge_index):
        edge_index = tlx.mask_select(edge_index, mask, axis = 1)
        edge_index = tlx.cast(edge_index, dtype = tlx.int64)
    elif check_is_numpy(edge_index):
        edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index, None
    else:
        edge_attr = tlx.mask_select(edge_attr, mask)
        return edge_index, edge_attr


def add_self_loops(
        edge_index, edge_attr=None, n_loops=1,
        fill_value: Union[float, str] = None,
        num_nodes: Optional[int] = None):
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted or has multi-dimensional edge features
    (:obj:`edge_attr != None`), edge features of self-loops will be added
    according to :obj:`fill_value`.
    
    .. code:: python

        >>> from gammagl.data import Graph
        >>> from gammagl.utils.loop import add_self_loops
        >>> import numpy
        >>> edge_index = tlx.constant([[0, 0, 0], [1, 2, 3]])
        array([[0, 0, 0],
                [1, 2, 3]])
        >>> edge_index, _ = add_self_loops(edge_index)
        array([[0, 0, 0, 0, 1, 2, 3],
               [1, 2, 3, 0, 1, 2, 3]])


    Parameters
    ----------
    edge_index: LongTensor
        The edge indices.
    n_loops: int
        the number of loops
    edge_attr: Tensor, optional
        Edge weights or multi-dimensional edge
        features. (default: :obj:`None`)
    fill_value: float or Tensor or str, optional
        The way to generate
        edge features of self-loops (in case :obj:`edge_attr != None`).
        If given as :obj:`float` or :class:`torch.Tensor`, edge features of
        self-loops will be directly given by :obj:`fill_value`.
        If given as :obj:`str`, edge features of self-loops are computed by
        aggregating all features of edges that point to the specific node,
        according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
        :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
    num_nodes: int, optional
        The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    Returns
    -------
    :class:`LongTensor`, :class:`Tensor`
    """
    N = maybe_num_nodes(edge_index, num_nodes)

    # loop_index = tlx.convert_to_tensor(np.arange(0, N), dtype=tlx.int64)
    # edge_index = tlx.convert_to_tensor(edge_index, dtype=tlx.int64) # torch raise Error
    loop_index = tlx.convert_to_tensor(np.arange(int(N)).repeat(n_loops), dtype=edge_index.dtype)
    loop_index = tlx.stack([loop_index, loop_index])

    if edge_attr is not None:
        if tlx.BACKEND in ['paddle']:
            shape = ([N] + edge_attr.shape[1:]) if edge_attr.ndim > 1 else (N,)
        else:
            shape = ([N] + tlx.get_tensor_shape(edge_attr)[1:]) if edge_attr.ndim > 1 else (N,)
        if fill_value is None:
            loop_attr = tlx.ones(shape, dtype=edge_attr.dtype)
        elif isinstance(fill_value, (int, float)):
            loop_attr = tlx.constant(value=fill_value, shape=shape, dtype=edge_attr.dtype)
        elif tlx.is_tensor(fill_value):
            loop_attr = tlx.convert_to_numpy(fill_value)
            if edge_attr.ndim != loop_attr.size:
                loop_attr = np.expand_dims(loop_attr, axis=0)

            # sizes = [N] + [1] * (loop_attr.size - 1)

            loop_attr = tlx.convert_to_tensor(np.repeat(loop_attr, [N], axis=0), dtype=fill_value.dtype)

        elif isinstance(fill_value, str):
            # TODO
            raise NotImplementedError
        #     loop_attr = scatter(edge_attr, edge_index[1], dim=0, dim_size=N,
        #                         reduce=fill_value)
        else:
            raise AttributeError("No valid 'fill_value' provided")

        edge_attr = tlx.concat([edge_attr, loop_attr], axis=0)

    edge_index = tlx.concat([edge_index, loop_index], axis=1)
    return edge_index, edge_attr
