# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/3/2

from typing import List
from gammagl.loader.link_loader import LinkLoader
from gammagl.sampler.neighbor_sampler import NeighborSampler


class LinkNeighborLoader(LinkLoader):
    r"""A link-based graph loader derived as an extension of the node-based
    :class:`gammagl.loader.NeighborLoader`.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    More specifically, this loader first selects a sample of edges from the
    set of input edges :obj:`edge_label_index` (which may or not be edges in
    the original graph) and then constructs a subgraph from all the nodes
    present in this list by sampling :obj:`num_neighbors` neighbors in each
    iteration.

    .. code-block:: python

        loader = LinkNeighborLoader(
            graph,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            edge_label_index=graph.edge_index,
        )

        sampled_graph = next(iter(loader))
        print(sampled_graph)

    It is additionally possible to provide edge labels for sampled edges, which
    are then added to the batch:

    .. code-block:: python

        loader = LinkNeighborLoader(
            graph,
            num_neighbors=[30] * 2,
            batch_size=128,
            edge_label_index=graph.edge_index,
            edge_label=torch.ones(graph.edge_index.size(1))
        )

        sampled_graph = next(iter(loader))
        print(sampled_graph)


    The rest of the functionality mirrors that of
    :class:`~gammagl.loader.NeighborLoader`, including support for
    heterogenous graphs.

    .. note::
        :obj:`neg_sampling_ratio` is currently implemented in an approximate
        way, *i.e.* negative edges may contain false negatives.

    Args:
        graph (gammagl.data.Data or gammagl.data.heterograph.HeteroGraph):
            The :class:`~gammagl.data.graph.Graph` or
            :class:`~gammagl.data.heterograph.HeteroGraph` graph object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample for each node in each iteration.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
            If an entry is set to :obj:`-1`, all neighbors will be included.
        edge_label_index (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
            The edge indices for which neighbors are sampled to create
            mini-batches.
            If set to :obj:`None`, all edges will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the edge type and corresponding edge indices.
            (default: :obj:`None`)
        edge_label (Tensor, optional): The labels of edge indices for
            which neighbors are sampled. Must be the same length as
            the :obj:`edge_label_index`. If set to :obj:`None` its set to
            `torch.zeros(...)` internally. (default: :obj:`None`)
        edge_label_time (Tensor, optional): The timestamps for edge indices
            for which neighbors are sampled. Must be the same length as
            :obj:`edge_label_index`. If set, temporal sampling will be
            used such that neighbors are guaranteed to fulfill temporal
            constraints, *i.e.*, neighbors have an earlier timestamp than
            the ouput edge. The :obj:`time_attr` needs to be set for this
            to work. (default: :obj:`None`)
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)
        neg_sampling_ratio (float, optional): The ratio of sampled negative
            edges to the number of positive edges.
            If :obj:`neg_sampling_ratio > 0` and in case :obj:`edge_label`
            does not exist, it will be automatically created and represents a
            binary classification task (:obj:`1` = edge, :obj:`0` = no edge).
            If :obj:`neg_sampling_ratio > 0` and in case :obj:`edge_label`
            exists, it has to be a categorical label from :obj:`0` to
            :obj:`num_classes - 1`.
            After negative sampling, label :obj:`0` represents negative edges,
            and labels :obj:`1` to :obj:`num_classes` represent the labels of
            positive edges.
            Note that returned labels are of type :obj:`torch.float` for binary
            classification (to facilitate the ease-of-use of
            :meth:`F.binary_cross_entropy`) and of type
            :obj:`torch.long` for multi-class classification (to facilitate the
            ease-of-use of :meth:`F.cross_entropy`). (default: :obj:`0.0`).
        time_attr (str, optional): The name of the attribute that denotes
            timestamps for the nodes in the graph. Only used if
            :obj:`edge_label_time` is set. (default: :obj:`None`)
        transform (Callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`edge_index` is sorted by column. This avoids internal
            re-sorting of the graph and can improve runtime and memory
            efficiency. (default: :obj:`False`)
        filter_per_worker (bool, optional): If set to :obj:`True`, will filter
            the returning graph in each worker's subprocess rather than in the
            main process.
            Setting this to :obj:`True` is generally not recommended:
            (1) it may result in too many open file handles,
            (2) it may slown down graph loading,
            (3) it requires operating on CPU tensors.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`tensorlayerx.dataflow.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last`.
    """

    def __init__(self, graph, num_neighbors: List[int], edge_label_index, edge_label, replace=False, directed=True,
                 neg_sampling_ratio=0.0, neighbor_sampler=None, is_sorted=False, **kwargs):
        edge_type = None

        if neighbor_sampler is None:
            neighbor_sampler = NeighborSampler(
                graph,
                num_neighbors=num_neighbors,
                replace=replace,
                directed=directed,
                input_type=edge_type,
                is_sorted=is_sorted
            )

        super(LinkNeighborLoader, self).__init__(graph=graph,
                                                 link_sampler=neighbor_sampler,
                                                 edge_label_index=edge_label_index,
                                                 edge_label=edge_label,
                                                 neg_sampling_ratio=neg_sampling_ratio,
                                                 **kwargs)
