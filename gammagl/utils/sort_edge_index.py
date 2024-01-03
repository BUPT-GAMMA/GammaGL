import tensorlayerx as tlx
from .num_nodes import maybe_num_nodes


def sort_edge_index(edge_index, edge_attr=None, num_nodes=None, sort_by_row: bool = True):
    """Row-wise sorts :obj:`edge_index`.
    
    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        sort_by_row (bool, optional): If set to :obj:`False`, will sort
            :obj:`edge_index` column-wise.

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :obj:`Tensor` or :obj:`List[Tensor]]`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = edge_index[1 - int(sort_by_row)] * num_nodes
    idx += edge_index[int(sort_by_row)]

    perm = tlx.ops.argsort(idx, descending=False)

    edge_index = tlx.gather(edge_index, indices=perm, axis=1)

    if edge_attr is None:
        return edge_index
    elif tlx.is_tensor(edge_attr):
        return edge_index, tlx.gather(edge_attr, indices=perm, axis=0)
    else:
        return edge_index, [tlx.gather(e, indices=perm, axis=0) for e in edge_attr]
