# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/3/27

from gammagl.loader.node_loader import NodeLoader
from gammagl.sampler.neighbor_sampler import NeighborSampler
from gammagl.loader.utils import get_input_nodes_index


class NodeNeighborLoader(NodeLoader):
    r"""A data loader that performs neighbor sampling as introduced in the
    `"Inductive Representation Learning on Large Graphs"
    <https://arxiv.org/abs/1706.02216>`_ paper.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    More specifically, :obj:`num_neighbors` denotes how much neighbors are
    sampled for each node in each iteration.
    :class:`~gammagl.loader.NodeNeighborLoader` takes in this list of
    :obj:`num_neighbors` and iteratively samples :obj:`num_neighbors[i]` for
    each node involved in iteration :obj:`i - 1`.

    Sampled nodes are sorted based on the order in which they were sampled.
    In particular, the first :obj:`batch_size` nodes represent the set of
    original mini-batch nodes.

    .. code-block:: python

        loader = NodeNeighborLoader(
            graph,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            input_nodes=data.train_mask,
        )

        sampled_data = next(iter(loader))
        print(sampled_data.batch_size)

    By default, the data loader will only include the edges that were
    originally sampled (:obj:`directed = True`).
    This option should only be used in case the number of hops is equivalent to
    the number of GNN layers.
    In case the number of GNN layers is greater than the number of hops,
    consider setting :obj:`directed = False`, which will include all edges
    between all sampled nodes (but is slightly slower as a result).

    Furthermore, :class:`~gammagl.loader.NodeNeighborLoader` works for both
    **homogeneous** graphs stored via :class:`~gammagl.data.graph.Graph` as
    well as **heterogeneous** graphs stored via
    :class:`~gammagl.data.heterograph.HeteroGraph`.
    When operating in heterogeneous graphs, up to :obj:`num_neighbors`
    neighbors will be sampled for each :obj:`edge_type`.
    However, more fine-grained control over
    the amount of sampled neighbors of individual edge types is possible:

    .. code-block:: python

        loader = NodeNeighborLoader(
            hetero_graph,
            # Sample 30 neighbors for each node and edge type for 2 iterations
            num_neighbors={key: [30] * 2 for key in hetero_graph.edge_types},
            # Use a batch size of 128 for sampling training nodes of type paper
            batch_size=128,
            input_nodes=('paper', hetero_graph['paper'].train_mask),
        )

        sampled_hetero_graph = next(iter(loader))
        print(sampled_hetero_graph['paper'].batch_size)

    .. not

    The :class:`~gammagl.loader.NeighborLoader` will return subgraphs
    where global node indices are mapped to local indices corresponding to this
    specific subgraph. However, often times it is desired to map the nodes of
    the current subgraph back to the global node indices. A simple trick to
    achieve this is to include this mapping as part of the :obj:`graph` object:

    .. code-block:: python

        # Assign each node its global node index:
        graph.n_id = tlx.arange(graph.num_nodes)

        loader = NeighborLoader(graph, ...)
        sampled_graph = next(iter(loader))
        print(sampled_graph.n_id)

    Args:
        graph (gammagl.data.graph.Graph or gammagl.data.heterograph.HeteroGraoh):
            The :class:`~gammagl.data.graph.Graph` or
            :class:`~gammagl.data.heterograph.HeteroGraoh` graph object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample for each node in each iteration.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
            If an entry is set to :obj:`-1`, all neighbors will be included.
        input_nodes (tlx Tensor or str or Tuple[str, tlx Tensor]): The
            indices of nodes for which neighbors are sampled to create
            mini-batches.
            If set to :obj:`None`, all nodes will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the node type and node indices. (default: :obj:`None`)
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`edge_index` is sorted by column. This avoids internal
            re-sorting of the graph and can improve runtime and memory
            efficiency. (default: :obj:`False`)
            Setting this to :obj:`True` is generally not recommended:
            (1) it may result in too many open file handles,
            (2) it may slown down graph loading,
            (3) it requires operating on CPU tensors.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`tensorlayerx.dataflow.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last`.
    """

    def __init__(self,
                 graph,
                 num_neighbors,
                 input_nodes_type=None,
                 replace: bool = False,
                 directed: bool = True,
                 is_sorted: bool = False,
                 neighbor_sampler=None,
                 **kwargs):
        node_type, _ = get_input_nodes_index(graph, input_nodes_type)
        if neighbor_sampler is None:
            neighbor_sampler = NeighborSampler(
                graph,
                num_neighbors=num_neighbors,
                replace=replace,
                directed=directed,
                input_type=node_type,
                is_sorted=is_sorted
            )

        super().__init__(
            graph=graph,
            node_sampler=neighbor_sampler,
            input_nodes_type=input_nodes_type,
            **kwargs,
        )
