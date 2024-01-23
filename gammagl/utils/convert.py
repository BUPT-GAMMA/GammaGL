import scipy.sparse as ssp
import tensorlayerx as tlx
import numpy as np

from .num_nodes import maybe_num_nodes


def to_scipy_sparse_matrix(edge_index, edge_attr = None, num_nodes = None):
    r"""Converts a graph given by edge indices and edge attributes to a scipy
    sparse matrix.

    Parameters
    ----------
        edge_index:
            The edge indices.
        edge_attr: tensor, optional
            Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes: int, optional
            The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    """
    row, col = tlx.convert_to_numpy(edge_index)
    
    if edge_attr is None:
        edge_attr = np.ones(row.shape[0])
    else:
        edge_attr = tlx.convert_to_numpy(tlx.reshape(edge_attr, (-1,)))
        assert edge_attr.shape[0] == row.shape[0]
    
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    return ssp.coo_matrix((edge_attr, (row, col)), (num_nodes, num_nodes))


def to_networkx(data, node_attrs = None, edge_attrs = None, graph_attrs = None,
    to_undirected = False, to_multi = False, remove_self_loops = False):
    r"""Converts a :class:`gammagl.data.Graph` instance to a
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data: A homogeneous or heterogeneous data object.
        node_attrs: The node attributes to be copied. (default: :obj:`None`)
        edge_attrs: The edge attributes to be copied. (default: :obj:`None`)
        graph_attrs: The graph attributes to be copied. (default: :obj:`None`)
        to_undirected : If set to :obj:`True`, will
            return a :class:`networkx.Graph` instead of a
            :class:`networkx.DiGraph`.
            By default, will include all edges and make them undirected.
            If set to :obj:`"upper"`, the undirected graph will only correspond
            to the upper triangle of the input adjacency matrix.
            If set to :obj:`"lower"`, the undirected graph will only correspond
            to the lower triangle of the input adjacency matrix.
            Only applicable in case the :obj:`data` object holds a homogeneous
            graph. (default: :obj:`False`)
        to_multi: if set to :obj:`True`, will return a
            :class:`networkx.MultiGraph` or a :class:`networkx:MultiDiGraph`
            (depending on the :obj:`to_undirected` option), which will not drop
            duplicated edges that may exist in :obj:`data`.
            (default: :obj:`False`)
        remove_self_loops: If set to :obj:`True`, will not
            include self-loops in the resulting graph. (default: :obj:`False`)

    Examples:
        >>> edge_index = tlx.convert_to_tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> data = Graph(edge_index=edge_index, num_nodes=4)
        >>> to_networkx(data)

    """
    import networkx as nx

    from gammagl.data import HeteroGraph

    to_undirected_upper: bool = to_undirected == 'upper'
    to_undirected_lower: bool = to_undirected == 'lower'

    to_undirected = to_undirected is True
    to_undirected |= to_undirected_upper or to_undirected_lower
    assert isinstance(to_undirected, bool)

    if isinstance(data, HeteroGraph) and to_undirected:
        raise ValueError("'to_undirected' is not supported in "
                         "'to_networkx' for heterogeneous graphs")

    if to_undirected:
        G = nx.MultiGraph() if to_multi else nx.Graph()
    else:
        G = nx.MultiDiGraph() if to_multi else nx.DiGraph()

    def to_networkx_value(value):
        return tlx.convert_to_numpy(value)
        # return value.tolist() if isinstance(value, Tensor) else value

    for key in graph_attrs or []:
        G.graph[key] = to_networkx_value(data[key])

    node_offsets = data.node_offsets
    for node_store in data.node_stores:
        start = node_offsets[node_store._key]
        assert node_store.num_nodes is not None
        for i in range(node_store.num_nodes):
            node_kwargs = {}
            if isinstance(data, HeteroGraph):
                node_kwargs['type'] = node_store._key
            for key in node_attrs or []:
                node_kwargs[key] = to_networkx_value(node_store[key][i])

            G.add_node(start + i, **node_kwargs)

    for edge_store in data.edge_stores:
        for i, (v, w) in enumerate(tlx.convert_to_numpy(tlx.transpose(edge_store.edge_index)).tolist()):
            if to_undirected_upper and v > w:
                continue
            elif to_undirected_lower and v < w:
                continue
            elif remove_self_loops and v == w and not edge_store.is_bipartite(
            ):
                continue

            edge_kwargs = {}
            if isinstance(data, HeteroGraph):
                v = v + node_offsets[edge_store._key[0]]
                w = w + node_offsets[edge_store._key[-1]]
                edge_kwargs['type'] = edge_store._key
            for key in edge_attrs or []:
                edge_kwargs[key] = to_networkx_value(edge_store[key][i])

            G.add_edge(v, w, **edge_kwargs)

    return G
