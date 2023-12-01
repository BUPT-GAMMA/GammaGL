import tensorlayerx as tlx
from gammagl.utils import sort_edge_index, coalesce
from .num_nodes import maybe_num_nodes


def is_undirected(
    edge_index, edge_attr=None, num_nodes=None) -> bool:
    r"""
        Returns :obj:`True` if the graph given by :attr:`edge_index` is
        undirected.
        
        Parameters
        ----------
        edge_index: tensor
            The edge indices.
        edge_attr: tensor, list[tensor], optional
            Edge weights or multi-
            dimensional edge features.
            If given as a list, will check for equivalence in all its entries.
            (default: :obj:`None`)
        num_nodes: int, optional
            The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

        Returns
        -------
        bool

    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_attr = [] if edge_attr is None else edge_attr
    edge_attr = [edge_attr] if tlx.is_tensor(edge_attr) else edge_attr

    edge_index1, edge_attr1 = sort_edge_index(
        edge_index,
        edge_attr,
        num_nodes=num_nodes,
        sort_by_row=True,
    )
    edge_index2, edge_attr2 = sort_edge_index(
        edge_index1,
        edge_attr1,
        num_nodes=num_nodes,
        sort_by_row=False,
    )

    return (bool(tlx.all(edge_index1[0] == edge_index2[1]))
            and bool(tlx.all(edge_index1[1] == edge_index2[0])) and all([
                tlx.all(e == e_T) for e, e_T in zip(edge_attr1, edge_attr2)
            ]))


def to_undirected(edge_index, edge_attr=None, num_nodes=None, reduce: str = "add"):
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.

    Parameters
    ----------
    edge_index: tensor
        The edge indices.
    edge_attr: tensor, list[tensor], optional
        Edge weights or multi-
        dimensional edge features.
        If given as a list, will remove duplicates for all its entries.
        (default: :obj:`None`)
    num_nodes: int, optional
        The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    reduce: str, optional
        The reduce operation to use for merging edge
        features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
        :obj:`"mul"`). (default: :obj:`"add"`)

    Returns
    -------
    :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else (:class:`LongTensor`, :obj:`Tensor` or :obj:`List[Tensor]]`)

    """
    # Maintain backward compatibility to `to_undirected(edge_index, num_nodes)`
    if isinstance(edge_attr, int):
        edge_attr = None
        num_nodes = edge_attr

    row, col = edge_index
    row, col = tlx.concat([row, col], axis=0), tlx.concat([col, row], axis=0)
    edge_index = tlx.stack([row, col], axis=0)

    if edge_attr is not None and tlx.is_tensor(edge_attr):
        edge_attr = tlx.concat([edge_attr, edge_attr], axis=0)
    elif edge_attr is not None:  # List[Tensor]
        edge_attr = [tlx.concat([e, e], axis=0) for e in edge_attr]

    return coalesce(edge_index, edge_attr, num_nodes, reduce)