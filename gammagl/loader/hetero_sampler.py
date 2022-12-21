from collections.abc import Sequence

import copy
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
from gammagl.data import HeteroGraph
from gammagl.data.storage import EdgeStorage, NodeStorage
import tensorlayerx as tlx
import numpy as np
from gammagl.typing import EdgeType
from gammagl.sample import hetero_sample

NumNeighbors = Union[List[int], Dict[EdgeType, List[int]]]

NumNeighbors = Union[List[int], Dict[EdgeType, List[int]]]


class NeighborSampler:
    def __init__(
            self,
            graph: HeteroGraph,
            num_neighbors: NumNeighbors,
            replace: bool = False,
            directed: bool = True,
            input_node_type: Optional[str] = None,
            share_memory: bool = False,
    ):
        self.data_cls = graph.__class__
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.directed = directed

        # Convert the graph data into a suitable format for sampling.
        # NOTE: Since C++ cannot take dictionaries with tuples as key as
        # input, edge type triplets are converted into single strings.
        out = to_hetero_csc(graph)
        self.colptr_dict, self.row_dict = out

        self.node_types, self.edge_types = graph.metadata()
        if isinstance(num_neighbors, (list, tuple)):
            num_neighbors = {key: num_neighbors for key in self.edge_types}
        assert isinstance(num_neighbors, dict)
        self.num_neighbors = {
            edge_type_to_str(key): value
            for key, value in num_neighbors.items()
        }

        self.num_hops = max([len(v) for v in self.num_neighbors.values()])

        assert isinstance(input_node_type, str)
        self.input_node_type = input_node_type


    def __call__(self, index):
        if not isinstance(index, np.ndarray):
            index = np.array(index, dtype=np.int64)

        if issubclass(self.data_cls, HeteroGraph):
            sample_fn = hetero_sample
            # for name in self.row_dict:
            #     self.row_dict[name] = self.row_dict[name].numpy()
            # for name in self.colptr_dict:
            #     self.colptr_dict[name] = self.colptr_dict[name].numpy()

            node_dict, row_dict, col_dict, edge_dict = sample_fn(
                self.node_types,
                self.edge_types,
                self.row_dict,
                self.colptr_dict,
                {self.input_node_type: index},
                self.num_neighbors,
                self.num_hops,
                self.replace,
                self.directed,
            )
            return node_dict, row_dict, col_dict, edge_dict, index.shape[0]


class Hetero_Neighbor_Sampler(tlx.dataflow.DataLoader):
    r"""A data loader that performs neighbor sampling as introduced in the
    `"Inductive Representation Learning on Large Graphs"
    <https://arxiv.org/abs/1706.02216>`_ paper.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    More specifically, :obj:`num_neighbors` denotes how much neighbors are
    sampled for each node in each iteration.
    :class:`~torch_geometric.loader.NeighborLoader` takes in this list of
    :obj:`num_neighbors` and iteratively samples :obj:`num_neighbors[i]` for
    each node involved in iteration :obj:`i - 1`.

    Sampled nodes are sorted based on the order in which they were sampled.
    In particular, the first :obj:`batch_size` nodes represent the set of
    original mini-batch nodes.

    .. code-block:: python

        from torch_geometric.datasets import Planetoid
        from torch_geometric.loader import NeighborLoader

        data = Planetoid(path, name='Cora')[0]

        loader = NeighborLoader(
            data,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            input_nodes=data.train_mask,
        )

        sampled_data = next(iter(loader))
        print(sampled_data.batch_size)
        >>> 128

    By default, the data loader will only include the edges that were
    originally sampled (:obj:`directed = True`).
    This option should only be used in case the number of hops is equivalent to
    the number of GNN layers.
    In case the number of GNN layers is greater than the number of hops,
    consider setting :obj:`directed = False`, which will include all edges
    between all sampled nodes (but is slightly slower as a result).

    Furthermore, :class:`~torch_geometric.loader.NeighborLoader` works for both
    **homogeneous** graphs stored via :class:`~torch_geometric.data.Data` as
    well as **heterogeneous** graphs stored via
    :class:`~torch_geometric.data.HeteroData`.
    When operating in heterogeneous graphs, more fine-grained control over
    the amount of sampled neighbors of individual edge types is possible, but
    not necessary:

    .. code-block:: python

        from torch_geometric.datasets import OGB_MAG
        from torch_geometric.loader import NeighborLoader

        hetero_data = OGB_MAG(path)[0]

        loader = NeighborLoader(
            hetero_data,
            # Sample 30 neighbors for each node and edge type for 2 iterations
            num_neighbors={key: [30] * 2 for key in hetero_data.edge_types},
            # Use a batch size of 128 for sampling training nodes of type paper
            batch_size=128,
            input_nodes=('paper', hetero_data['paper'].train_mask),
        )

        sampled_hetero_data = next(iter(loader))
        print(sampled_hetero_data['paper'].batch_size)
        >>> 128

    .. note::

        For an example of using
        :class:`~torch_geometric.loader.NeighborLoader`, see
        `examples/hetero/to_hetero_mag.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py>`_.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample for each node in each iteration.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
            If an entry is set to :obj:`-1`, all neighbors will be included.
        input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
            indices of nodes for which neighbors are sampled to create
            mini-batches.
            Needs to be either given as a :obj:`torch.LongTensor` or
            :obj:`torch.BoolTensor`.
            If set to :obj:`None`, all nodes will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the node type and node indices. (default: :obj:`None`)
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)
        transform (Callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """

    def __init__(
            self,
            graph: HeteroGraph,
            num_neighbors: NumNeighbors,
            input_nodes,
            replace: bool = False,
            directed: bool = True,
            neighbor_sampler: Optional[NeighborSampler] = None,
            **kwargs,
    ):

        self.graph = graph

        # Save for PyTorch Lightning < 1.6:
        self.num_neighbors = num_neighbors
        self.input_nodes = input_nodes
        self.replace = replace
        self.directed = directed
        self.neighbor_sampler = neighbor_sampler

        if neighbor_sampler is None:
            input_node_type = get_input_node_type(input_nodes)
            self.neighbor_sampler = NeighborSampler(
                graph, num_neighbors, replace, directed, input_node_type,
                share_memory=kwargs.get('num_workers', 0) > 0)

        return super().__init__(get_input_node_indices(graph, input_nodes),
                                collate_fn=self.neighbor_sampler, **kwargs)

    def transform_fn(self, out: Any) -> HeteroGraph:
        # TODO: 增加一個顯示batch_size大小的東西
        node_dict, row_dict, col_dict, edge_dict, batch_size = out
        data = filter_hetero_data(self.graph, node_dict, row_dict, col_dict,
                                  edge_dict)
        # TODO：这里将data的edge_stores 变成tensor!这样就可以省去很多麻烦
        for values in data.edge_stores:
            for key in values:
                values[key] = tlx.convert_to_tensor(values[key], tlx.int64)
        data[self.neighbor_sampler.input_node_type].batch_size = batch_size
        return data

    def _get_iterator(self) -> Iterator:
        return DataLoaderIterator(super()._get_iterator(), self.transform_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


###############################################################################



def get_input_node_type(input_nodes) -> Optional[str]:
    if isinstance(input_nodes, str):
        return input_nodes
    if isinstance(input_nodes, (list, tuple)):
        assert isinstance(input_nodes[0], str)
        return input_nodes[0]
    return None


def get_input_node_indices(graph: HeteroGraph,
                           input_nodes):
    if isinstance(graph, HeteroGraph):
        if isinstance(input_nodes, str):
            input_nodes = (input_nodes, None)
        assert isinstance(input_nodes, (list, tuple))
        assert len(input_nodes) == 2
        assert isinstance(input_nodes[0], str)
        if input_nodes[1] is None:
            return range(graph[input_nodes[0]].num_nodes)
        input_nodes = input_nodes[1]

    if not isinstance(input_nodes, np.ndarray):
        input_nodes = tlx.convert_to_numpy(input_nodes).tolist()


    return input_nodes

def edge_type_to_str(edge_type) -> str:
    # Since C++ cannot take dictionaries with tuples as key as input, edge type
    # triplets need to be converted into single strings.
    return edge_type if isinstance(edge_type, str) else '__'.join(edge_type)

def filter_hetero_data(
    graph: HeteroGraph,
    node_dict: Dict[str, np.ndarray],
    row_dict: Dict[str, np.ndarray],
    col_dict: Dict[str, np.ndarray],
    edge_dict: Dict[str, np.ndarray],
):
    # Filters a heterogeneous data object to only hold nodes in `node` and
    # edges in `edge` for each node and edge type, respectively:
    out = copy.copy(graph)

    for node_type in graph.node_types:
        filter_node_store_(graph[node_type], out[node_type],
                           node_dict[node_type])

    for edge_type in graph.edge_types:
        edge_type_str = edge_type_to_str(edge_type)
        filter_edge_store_(graph[edge_type], out[edge_type],
                           row_dict[edge_type_str], col_dict[edge_type_str],
                           edge_dict[edge_type_str])

    return out

def filter_node_store_(store: NodeStorage, out_store: NodeStorage,
                       index):
    # Filters a node storage object to only hold the nodes in `index`:
    for key, value in store.items():
        if key == 'num_nodes':
            out_store.num_nodes = len(index)
        elif store.is_node_attr(key):
            if tlx.BACKEND == 'torch':
                index = tlx.convert_to_tensor(index, tlx.int64).to(value.device)
            else:
                index = tlx.convert_to_tensor(index, tlx.int64)
            out_store[key] = tlx.gather(value, index)
    return store


def filter_edge_store_(store: EdgeStorage, out_store: EdgeStorage, row,
                       col, index):
    # Filters a edge storage object to only hold the edges in `index`,
    # which represents the new graph as denoted by `(row, col)`:
    for key, value in store.items():
        if key == 'edge_index':
            edge_index = np.stack([row, col], axis=0)
            out_store.edge_index = edge_index
        elif store.is_edge_attr(key):
            out_store[key] = value[index]
    return store



def to_hetero_csc(hetero_graph):
    # Convert the heterogeneous graph data into a suitable format for sampling
    # (CSC format).
    # Returns dictionaries holding `colptr` and `row` indices as well as edge
    # permutations for each edge type, respectively.
    # Since C++ cannot take dictionaries with tuples as key as input, edge type
    # triplets are converted into single strings.
    colptr_dict, row_dict = {}, {}

    for store in hetero_graph.edge_stores:
        key = edge_type_to_str(store._key)


        if hasattr(store, 'edge_index'):
            # ind = np.argsort(edge_index[1], axis=0)
            # edge_index = np.array(edge_index.T[ind]).T
            size = store.size()
            edge = store.edge_index
            perm = np.argsort(np.add(edge[1] * size[0], edge[0]))
            edge = tlx.gather(edge, perm, 1)
            indptr = np.concatenate(([0], np.bincount(edge[1]).cumsum()))
            if size[1] > indptr.shape[0]:
                cur = size[1] - indptr.shape[0]
                indptr = np.concatenate((indptr, np.full(shape=(cur + 1,), fill_value=indptr[-1], dtype=np.int64)))
            colptr_dict[key], row_dict[key] = indptr, tlx.convert_to_numpy(edge[0])
            # (row, col) = store.edge_index
            # perm = np.argsort(np.add(col * size[0], row))
            # col = col[perm]
            # colptr = [0]
            # row = row[perm]

        else:
            raise AttributeError("HeteroGraph object does not contain attributes "
                                 "'edge_index'")
    return colptr_dict, row_dict



from tensorlayerx.dataflow.utils import _BaseDataLoaderIter
class DataLoaderIterator(object):
    def __init__(self, iterator: _BaseDataLoaderIter, transform_fn: Callable):
        self.iterator = iterator
        self.transform_fn = transform_fn

    def __iter__(self) -> 'DataLoaderIterator':
        return self

    def _reset(self, loader: Any, first_iter: bool = False):
        self.iterator._reset(loader, first_iter)

    def __len__(self) -> int:
        return len(self.iterator)

    def __next__(self) -> Any:
        return self.transform_fn(next(self.iterator))
