# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/3/2

from tensorlayerx.dataflow import DataLoader, Dataset
import tensorlayerx as tlx
from gammagl.loader.utils import DataLoaderIter, filter_graph
# Dataset based on edges, support multi-type edges
from gammagl.sampler.neighbor_sampler import SamplerOutput


class LinkDataset(Dataset):
    def __init__(self, edge_label_index, edge_label):
        super().__init__()
        self.edge_label_index = edge_label_index
        self.edge_label = edge_label

    def __getitem__(self, idx: int):
        return (
            self.edge_label_index[0, idx],
            self.edge_label_index[1, idx],
            self.edge_label[idx],
        )

    def __len__(self) -> int:
        return self.edge_label_index.shape[1]


class LinkLoader(DataLoader):
    r"""A graph loader that performs neighbor sampling from link information,
    using a generic :class:`~gammagl.sampler.BaseSampler`
    implementation that defines a :meth:`sample_from_edges` function and is
    supported on the provided input :obj:`graph` object.

    Args:
        graph (gammagl.data.graph.Graph or gammagl.data.heterograph.HeteroGraph):
            The :class:`~gammagl.data.graph.Graph` or
            :class:`~gammagl.data.heterograph.HeteroGraph` graph object.
        link_sampler (gammagl.sampler.BaseSampler): The sampler
            implementation to be used with this loader. Note that the
            sampler implementation must be compatible with the input graph
            object.
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
            `tlx.zeros(...)` internally. (default: :obj:`None`)
        neg_sampling_ratio (float, optional): the number of negative samples
            to include as a ratio of the number of positive examples
            (default: 0).
        **kwargs (optional): Additional arguments of
            :class:`tensorlayerx.dataflow.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last`.
    """

    def __init__(self, graph, link_sampler, edge_label_index=None, edge_label=None, neg_sampling_ratio=0.0, **kwargs):
        self.graph = graph
        self.link_sampler = link_sampler
        self.edge_label_index = edge_label_index
        self.edge_label = edge_label
        self.neg_sampling_ratio = neg_sampling_ratio

        if self.edge_label_index is None:
            edge_label_index = graph.edge_index
        if self.edge_label is None:
            # edge_label = tlx.zeros(edge_label_index.size(1))
            edge_label = tlx.zeros(edge_label_index.shape[1])

        super(LinkLoader, self).__init__(LinkDataset(edge_label_index, edge_label),
                                         collate_fn=self.collate_fn,
                                         **kwargs)

    def collate_fn(self, index):
        out = self.link_sampler.sample_from_edges(
            index,
            negative_sampling_ratio=self.neg_sampling_ratio,
        )
        return out

    def filter_fn(self, out: SamplerOutput):

        edge_label_index, edge_label = out.metadata
        graph = filter_graph(self.graph, out.node, out.row, out.col, out.edge,
                             self.link_sampler.edge_permutation)
        graph.edge_label_index = edge_label_index
        graph.edge_label = edge_label

        return graph

    def _get_iterator(self):
        return DataLoaderIter(super(LinkLoader, self)._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
