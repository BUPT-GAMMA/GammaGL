# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/3/27

from tensorlayerx.dataflow import DataLoader
import tensorlayerx as tlx
from typing import Union

from gammagl.data import HeteroGraph
from gammagl.loader.utils import filter_graph, filter_hetero_graph, DataLoaderIter
from gammagl.sampler.neighbor_sampler import SamplerOutput, HeteroSamplerOutput
from gammagl.loader.utils import get_input_nodes_index


class NodeLoader(DataLoader):
    r"""A graph loader that performs neighbor sampling from node information,
    using a generic :class:`~gammagl.sampler.BaseSampler`
    implementation that defines a :meth:`sample_from_nodes` function and is
    supported on the provided input :obj:`data` object.

    Parameters
    ----------
    data: graph, heterograph
        The :class:`~gammagl.data.Graph` or
        :class:`~gammagl.data.HeteroGraph` graph object.
    node_sampler: sampler
        The sampler implementation to be used with this loader. Note that the
        sampler implementation must be compatible with the input data
        object.
    input_nodes: tensor, str, tuple[str, tensor]
        The indices of nodes for which neighbors are sampled to create
        mini-batches.
        If set to :obj:`None`, all nodes will be considered.
        In heterogeneous graphs, needs to be passed as a tuple that holds
        the node type and node indices. (default: :obj:`None`)
    transform: callable, optional
        A function/transform that takes in
        a sampled mini-batch and returns a transformed version.
        (default: :obj:`None`)
    filter_per_worker: bool, optional
        If set to :obj:`True`, will filter
        the returning data in each worker's subprocess rather than in the
        main process.
        Setting this to :obj:`True` is generally not recommended:
        (1) it may result in too many open file handles,
        (2) it may slown down data loading,
        (3) it requires operating on CPU tensors.
        (default: :obj:`False`)
    **kwargs: optional
        Additional arguments of
        :class:`tensorlayerx.dataflow.DataLoader`, such as :obj:`batch_size`,
        :obj:`shuffle`, :obj:`drop_last`.

    """

    def __init__(
            self,
            graph,
            node_sampler,
            input_nodes_type=None,
            **kwargs,
    ):
        self.graph = graph
        self.node_sampler = node_sampler
        self.input_nodes = input_nodes_type

        node_type, input_nodes_index = get_input_nodes_index(graph, input_nodes_type)
        self.input_type = node_type

        super().__init__(input_nodes_index, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, index):
        if isinstance(index, (tuple, list)):
            index = tlx.convert_to_tensor(index)

        out = self.node_sampler.sample_from_nodes(index)

        return out

    def filter_fn(self, out: Union[SamplerOutput, HeteroSamplerOutput]):

        if isinstance(out, SamplerOutput):
            graph = filter_graph(self.graph, out.node, out.row, out.col, out.edge,
                                 self.node_sampler.edge_permutation)
            graph.batch = out.batch
            graph.batch_size = out.metadata
        elif isinstance(out, HeteroSamplerOutput):
            if isinstance(self.graph, HeteroGraph):
                graph = filter_hetero_graph(self.graph, out.node, out.row,
                                            out.col, out.edge,
                                            self.node_sampler.edge_permutation)

            for key, batch in (out.batch or {}).items():
                graph[key].batch = batch
            graph[self.input_type].batch_size = out.metadata

        else:
            raise TypeError("Not supported custom type.")

        return graph

    def _get_iterator(self):
        return DataLoaderIter(super()._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
