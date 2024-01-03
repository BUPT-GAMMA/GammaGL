from typing import List

import tensorlayerx as tlx
from gammagl.data import HeteroGraph
from gammagl.transforms import BaseTransform
from gammagl.typing import EdgeType


class AddMetaPaths(BaseTransform):
    r""" Adds additional edge types to a
    :class:`~gammagl.data.HeteroGraph` object between the source node
    type and the destination node type of a given :obj:`metapath`, as described
    in the `"Heterogenous Graph Attention Networks"
    <https://arxiv.org/abs/1903.07293>`_ paper.
    Meta-path based neighbors can exploit different aspects of structure
    information in heterogeneous graphs.
    Formally, a metapath is a path of the form

    .. math::
        \mathcal{V}_1 \xrightarrow{R_1} \mathcal{V}_2 \xrightarrow{R_2} \ldots
        \xrightarrow{R_{\ell-1}} \mathcal{V}_{\ell}

    in which :math:`\mathcal{V}_i` represents node types, and :math:`R_j`
    represents the edge type connecting two node types.
    The added edge type is given by the sequential multiplication  of
    adjacency matrices along the metapath, and is added to the
    :class:`~gammagl.data.HeteroGraph` object as edge type
    :obj:`(src_node_type, "metapath_*", dst_node_type)`, where
    :obj:`src_node_type` and :obj:`dst_node_type` denote :math:`\mathcal{V}_1`
    and :math:`\mathcal{V}_{\ell}`, repectively.
    In addition, a :obj:`metapath_dict` object is added to the
    :class:`~gammagl.data.HeteroGraph` object which maps the
    metapath-based edge type to its original metapath.

    .. code:: python

        >>> from gammagl.datasets import DBLP
        >>> from gammagl.data import HeteroGraph
        >>> from gammagl.transforms import AddMetaPaths
        >>> data = DBLP(root)[0]
        # 4 node types: "paper", "author", "conference", and "term"
        # 6 edge types: ("paper","author"), ("author", "paper"),
        #               ("paper, "term"), ("paper", "conference"),
        #               ("term, "paper"), ("conference", "paper")
        # Add two metapaths:
        # 1. From "paper" to "paper" through "conference"
        # 2. From "author" to "conference" through "paper"
        >>> metapaths = [[("paper", "conference"), ("conference", "paper")],
                     [("author", "paper"), ("paper", "conference")]]
        >>> data = AddMetaPaths(metapaths)(data)
        >>> print(data.edge_types)
        [("author", "to", "paper"), ("paper", "to", "author"),
             ("paper", "to", "term"), ("paper", "to", "conference"),
             ("term", "to", "paper"), ("conference", "to", "paper"),
             ("paper", "metapath_0", "paper"),
             ("author", "metapath_1", "conference")]
        >>> print(data.metapath_dict)
        {("paper", "metapath_0", "paper"): [("paper", "conference"),
                                                ("conference", "paper")],
             ("author", "metapath_1", "conference"): [("author", "paper"),
                                                      ("paper", "conference")]}

    Parameters
    ----------
    metapaths: List[List[Tuple[str, str, str]]]
        The metapaths described
        by a list of lists of
        :obj:`(src_node_type, rel_type, dst_node_type)` tuples.
    drop_orig_edges: bool, optional
        If set to :obj:`True`, existing edge
        types will be dropped. (default: :obj:`False`)
    keep_same_node_type: bool, optional
        If set to :obj:`True`, existing
        edge types between the same node type are not dropped even in case
        :obj:`drop_orig_edges` is set to :obj:`True`.
        (default: :obj:`False`)
    drop_unconnected_nodes: bool, optional
        If set to :obj:`True` drop
        node types not connected by any edge type. (default: :obj:`False`)
    """
    def __init__(self, metapaths: List[List[EdgeType]],
                 drop_orig_edges: bool = False,
                 keep_same_node_type: bool = False,
                 drop_unconnected_nodes: bool = False):

        for path in metapaths:
            assert len(path) >= 2, f"Invalid metapath '{path}'"
            assert all([
                j[-1] == path[i + 1][0] for i, j in enumerate(path[:-1])
            ]), f"Invalid sequence of node types in '{path}'"

        self.metapaths = metapaths
        self.drop_orig_edges = drop_orig_edges
        self.keep_same_node_type = keep_same_node_type
        self.drop_unconnected_nodes = drop_unconnected_nodes

    def __call__(self, data: HeteroGraph) -> HeteroGraph:
        edge_types = data.edge_types  # save original edge types
        data.metapath_dict = {}

        for j, metapath in enumerate(self.metapaths):
            for edge_type in metapath:
                assert data._to_canonical(
                    edge_type) in edge_types, f"'{edge_type}' not present"

            # edge_type = metapath[0]
            # adj1 = SparseTensor.from_edge_index(
            #     edge_index=data[edge_type].edge_index,
            #     sparse_sizes=data[edge_type].size())
            adj = 1
            for i, edge_type in enumerate(metapath):
                # adj2 = SparseTensor.from_edge_index(
                #     edge_index=data[edge_type].edge_index,
                #     sparse_sizes=data[edge_type].size())
                adj = adj * data.adj(scipy_fmt='csr', etype=edge_type)

            adj_coo = adj.tocoo()
            row = tlx.convert_to_tensor(adj_coo.row, dtype = tlx.int64)
            col = tlx.convert_to_tensor(adj_coo.col, dtype = tlx.int64)
            new_edge_type = (metapath[0][0], f'metapath_{j}', metapath[-1][-1])
            data[new_edge_type].edge_index = tlx.stack([row, col])
            data.metapath_dict[new_edge_type] = metapath

        if self.drop_orig_edges:
            for i in edge_types:
                if self.keep_same_node_type and i[0] == i[-1]:
                    continue
                else:
                    del data[i]

        # remove nodes not connected by any edge type.
        if self.drop_unconnected_nodes:
            new_edge_types = data.edge_types
            node_types = data.node_types
            connected_nodes = set()
            for i in new_edge_types:
                connected_nodes.add(i[0])
                connected_nodes.add(i[-1])
            for node in node_types:
                if node not in connected_nodes:
                    del data[node]

        return data