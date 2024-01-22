import warnings
import copy
import numpy as np
import tensorlayerx as tlx

from collections.abc import Mapping, Sequence
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union
)
from gammagl.data.storage import (
    BaseStorage,
    EdgeStorage,
    GlobalStorage,
    NodeStorage
)

from gammagl.sparse.sparse_adj import CSRAdj
from gammagl.utils.loop import add_self_loops
from gammagl.utils.check import check_is_numpy
from gammagl.typing import NodeType, EdgeType


class BaseGraph:
    r"""
    """

    def __getattr__(self, key: str) -> Any:
        raise NotImplementedError

    def __setattr__(self, key: str, value: Any):
        raise NotImplementedError

    def __delattr__(self, key: str):
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        raise NotImplementedError

    def __setitem__(self, key: str, value: Any):
        raise NotImplementedError

    def __delitem__(self, key: str):
        raise NotImplementedError

    def __copy__(self):
        raise NotImplementedError

    def __deepcopy__(self, memo):
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def stores_as(self, data: 'BaseGraph'):
        raise NotImplementedError

    @property
    def stores(self) -> List[BaseStorage]:
        raise NotImplementedError

    @property
    def node_stores(self) -> List[NodeStorage]:
        raise NotImplementedError

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        r"""Returns a dictionary of stored key/value pairs."""
        raise NotImplementedError

    def to_namedtuple(self) -> NamedTuple:
        r"""Returns a :obj:`NamedTuple` of stored key/value pairs."""
        raise NotImplementedError

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        r"""Returns the dimension for which the value :obj:`value` of the
        attribute :obj:`key` will get concatenated when creating mini-batches
        using :class:`gammagl.loader.DataLoader`.

        .. note::
            This method is for internal use only, and should only be overridden
            in case the mini-batch creation process is corrupted for a specific
            attribute.
        """
        raise NotImplementedError

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        r"""Returns the incremental count to cumulatively increase the value
        :obj:`value` of the attribute :obj:`key` when creating mini-batches
        using :class:`gammagl.loader.DataLoader`.

        .. note::
            This method is for internal use only, and should only be overridden
            in case the mini-batch creation process is corrupted for a specific
            attribute.
        """
        raise NotImplementedError

    def debug(self):
        raise NotImplementedError

    ###########################################################################

    @property
    def keys(self) -> List[str]:
        r"""Returns a list of all graph attribute names."""
        out = []
        for store in self.stores:
            out += list(store.keys())
        return list(set(out))

    def __len__(self) -> int:
        r"""Returns the number of graph attributes."""
        return len(self.keys)

    def __contains__(self, key: str) -> bool:
        r"""Returns :obj:`True` if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __getstate__(self) -> Dict[str, Any]:
        # func `__getstate__` solve the problem of pickle Graph
        # https://stackoverflow.com/questions/50156118/recursionerror-maximum-recursion-depth-exceeded-while-calling-a-python-object-w
        return self.__dict__

    def __setstate__(self, mapping: Dict[str, Any]):
        for key, value in mapping.items():
            self.__dict__[key] = value

    @property
    def num_nodes(self) -> Optional[int]:
        r"""Returns the number of nodes in the graph.

        .. note::
            The number of nodes in the data object is automatically inferred
            in case node-level attributes are present, *e.g.*, :obj:`data.x`.
            In some cases, however, a graph may only be given without any
            node-level attributes.
            GammaGL then *guesses* the number of nodes according to
            :obj:`edge_index.max().item() + 1`.
            However, in case there exists isolated nodes, this number does not
            have to be correct which can result in unexpected behaviour.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`graph.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        try:
            return sum([v.num_nodes for v in self.node_stores])
        except TypeError:
            return None

    def size(
            self, dim: Optional[int] = None
    ) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:
        r"""Returns the size of the adjacency matrix induced by the graph."""
        size = (self.num_nodes, self.num_nodes)
        return size if dim is None else size[dim]

    @property
    def num_edges(self) -> int:
        r"""Returns the number of edges in the graph.
        For undirected graphs, this will return the number of bi-directional
        edges, which is double the amount of unique edges."""
        return sum([v.num_edges for v in self.edge_stores])
    
    @property
    def node_offsets(self) -> Dict[NodeType, int]:
        out: Dict[NodeType, int] = {}
        offset: int = 0
        for store in self.node_stores:
            out[store._key] = offset
            offset += store.num_nodes
        return out

    def is_coalesced(self) -> bool:
        r"""Returns :obj:`True` if the :obj:`edge_index` is sorted
        and does not contain duplicate entries."""
        return all([store.is_coalesced() for store in self.edge_stores])

    def coalesce(self):
        r"""Sorts and removes duplicated entries from edge indices
        :obj:`edge_index`."""
        for store in self.edge_stores:
            store.coalesce()
        return self

    def has_isolated_nodes(self) -> bool:
        r"""Returns :obj:`True` if the graph contains isolated nodes."""
        return any([store.has_isolated_nodes() for store in self.edge_stores])

    def has_self_loops(self) -> bool:
        """Returns :obj:`True` if the graph contains self-loops."""
        return any([store.has_self_loops() for store in self.edge_stores])

    # The Graph is directed in GammaGL.

    def is_undirected(self) -> bool:
        r"""Returns :obj:`True` if graph edges are undirected."""
        return all([store.is_undirected() for store in self.edge_stores])

    def is_directed(self) -> bool:
        r"""Returns :obj:`True` if graph edges are directed."""
        return not self.is_undirected()

    def apply_(self, func: Callable, *args: List[str]):
        r"""Applies the in-place function :obj:`func`, either to all attributes
        or only the ones given in :obj:`*args`."""
        for store in self.stores:
            store.apply_(func, *args)
        return self

    def apply(self, func: Callable, *args: List[str]):
        r"""Applies the function :obj:`func`, either to all attributes or only
        the ones given in :obj:`*args`."""
        for store in self.stores:
            store.apply(func, *args)
        return self

    def clone(self, *args: List[str]):
        r"""Performs cloning of tensors, either for all attributes or only the
        ones given in :obj:`*args`.
        It can only be used in Torch.
        """
        return copy.copy(self).apply(lambda x: x.clone(), *args)

    def _apply_to_tensor(self, key, value, inplace=True):
        if value is None:
            return value

        if isinstance(value, CSRAdj):
            value = value.tensor(inplace=inplace)

        elif isinstance(value, dict):
            if inplace:
                for k, v in value.items():
                    value[k] = tlx.convert_to_tensor(v)
            else:
                new_value = {}
                for k, v in value.items():
                    new_value[k] = tlx.convert_to_tensor(v)
                value = new_value
        else:
            if tlx.is_tensor(value):
                pass
            elif check_is_numpy(value):
                if key in ['edge_index', 'y', 'edge_type', 'train_idx', 'test_idx', 'train_y']:
                    value = tlx.convert_to_tensor(value, dtype=tlx.int64)
                elif key in ['train_mask', 'val_mask', 'test_mask', ]:
                    value = tlx.convert_to_tensor(value, dtype=tlx.bool)
                else:
                    value = tlx.convert_to_tensor(value, dtype=tlx.float32)
        return value

    def _apply_to_numpy(self, key, value, inplace=True):
        if value is None:
            return value

        if isinstance(value, CSRAdj):
            value = value.numpy(inplace=inplace)
        elif isinstance(value, dict):
            if inplace:
                for k, v in value.items():
                    value[k] = v.numpy()
            else:
                new_value = {}
                for k, v in value.items():
                    new_value[k] = v.numpy()
                value = new_value
        else:
            if check_is_numpy(value):
                pass
            elif tlx.is_tensor(value):
                # can't assign type of numpy
                value = tlx.convert_to_numpy(value)
        return value


# def contiguous(self, *args: List[str]):
# 	r"""Ensures a contiguous memory layout, either for all attributes or
# 	only the ones given in :obj:`*args`."""
# 	return self.apply(lambda x: x.contiguous(), *args)
#
# def to(self, device: Union[int, str], *args: List[str],
# 	   non_blocking: bool = False):
# 	r"""Performs tensor device conversion, either for all attributes or
# 	only the ones given in :obj:`*args`."""
# 	return self.apply(
# 		lambda x: x.to(device=device, non_blocking=non_blocking), *args)
#
# def cpu(self, *args: List[str]):
# 	r"""Copies attributes to CPU memory, either for all attributes or only
# 	the ones given in :obj:`*args`."""
# 	return self.apply(lambda x: x.cpu(), *args)
#
# def cuda(self, device: Optional[Union[int, str]] = None, *args: List[str],
# 		 non_blocking: bool = False):
# 	r"""Copies attributes to CUDA memory, either for all attributes or only
# 	the ones given in :obj:`*args`."""
# 	# Some PyTorch tensor like objects require a default value for `cuda`:
# 	device = 'cuda' if device is None else device
# 	return self.apply(lambda x: x.cuda(device, non_blocking=non_blocking),
# 					  *args)
#
# def pin_memory(self, *args: List[str]):
# 	r"""Copies attributes to pinned memory, either for all attributes or
# 	only the ones given in :obj:`*args`."""
# 	return self.apply(lambda x: x.pin_memory(), *args)
#
# def share_memory_(self, *args: List[str]):
# 	r"""Moves attributes to shared memory, either for all attributes or
# 	only the ones given in :obj:`*args`."""
# 	return self.apply_(lambda x: x.share_memory_(), *args)
#
# def detach_(self, *args: List[str]):
# 	r"""Detaches attributes from the computation graph, either for all
# 	attributes or only the ones given in :obj:`*args`."""
# 	return self.apply_(lambda x: x.detach_(), *args)
#
# def detach(self, *args: List[str]):
# 	r"""Detaches attributes from the computation graph by creating a new
# 	tensor, either for all attributes or only the ones given in
# 	:obj:`*args`."""
# 	return self.apply(lambda x: x.detach(), *args)
#
# def requires_grad_(self, *args: List[str], requires_grad: bool = True):
# 	r"""Tracks gradient computation, either for all attributes or only the
# 	ones given in :obj:`*args`."""
# 	return self.apply_(
# 		lambda x: x.requires_grad_(requires_grad=requires_grad), *args)
#
# def record_stream(self, stream: torch.cuda.Stream, *args: List[str]):
# 	r"""Ensures that the tensor memory is not reused for another tensor
# 	until all current work queued on :obj:`stream` has been completed,
# 	either for all attributes or only the ones given in :obj:`*args`."""
# 	return self.apply_(lambda x: x.record_stream(stream), *args)
#
# @property
# def is_cuda(self) -> bool:
# 	r"""Returns :obj:`True` if any :class:`torch.Tensor` attribute is
# 	stored on the GPU, :obj:`False` otherwise."""
# 	for store in self.stores:
# 		for value in store.values():
# 			if isinstance(value, Tensor) and value.is_cuda:
# 				return True
# 	return False


class Graph(BaseGraph):
    r"""
        A Graph object describe a homogeneous graph. The graph object
        will hold node-level, link-level and graph-level attributes. In
        general, :class:`~gammagl.data.Graph` tries to mimic the behaviour
        of a regular Python dictionary. In addition, it provides useful
        functionality for analyzing graph structures, and provides basic
        tensor functionalities.

        .. code:: python

            >>> from gammagl.data import Graph
            >>> import numpy
            >>> g = Graph(x=numpy.random.randn(5, 16), edge_index=[[0, 0, 0], [1, 2, 3]], num_nodes=5,)
            >>> print(g)
            GNN Graph instance.
            number of nodes: 5
            number of edges: 2

            >>> print(g.indegree.numpy(), g.outdegree.numpy())
            [0. 1. 1. 1. 0.] [3. 0. 0. 0. 0.]

        Parameters
        ----------
        x: tensor, list, numpy.array
            Node feature matrix with shape :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
        edge_index: tensor, list, numpy.array
            Graph connectivity in COO format with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr: tensor
            Edge feature matrix with shape :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        num_nodes: int
            The specified number of nodes. (default: :obj:`None`)
        y: tensor
            Graph-level or node-level ground-truth labels with arbitrary shape. (default: :obj:`None`)
        to_tensor: bool
            Set data to tensor
        spr_format: list(str)
            Specify the other sparse storage format, like `csc` and `csr`. (default: :obj:`None`)

    """

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, spr_format=None, to_tensor=True,
                 **kwargs):
        self.__dict__['_store'] = GlobalStorage(_parent=self)
        if edge_index is not None:
            self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        # self.num_nodes = num_nodes
        for key, value in kwargs.items():
            setattr(self, key, value)
        if to_tensor:
            self.tensor()
        else:
            self.numpy()
        if spr_format is not None:
            if 'csr' in spr_format:
                self.csr_adj = CSRAdj.from_edges(self.edge_index[0], self.edge_index[1], self.num_nodes)
            if 'csc' in spr_format:
                self.csc_adj = CSRAdj.from_edges(self.edge_index[1], self.edge_index[0], self.num_nodes)

    def __getattr__(self, key: str) -> Any:
        # Called when the default attribute access fails, which means getattr
        return getattr(self._store, key)

    def __setattr__(self, key: str, value: Any):
        setattr(self._store, key, value)

    def __delattr__(self, key: str):
        delattr(self._store, key)

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value: Any):
        self._store[key] = value

    def __delitem__(self, key: str):
        if key in self._store:
            del self._store[key]

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__['_store'] = copy.copy(self._store)
        out._store._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        out._store._parent = out
        return out

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        has_dict = any([isinstance(v, Mapping) for v in self._store.values()])

        if not has_dict:
            info = [size_repr(k, v) for k, v in self._store.items()]
            info = ', '.join(info)
            return f'{cls}({info})'
        else:
            info = [size_repr(k, v, indent=2) for k, v in self._store.items()]
            info = ',\n'.join(info)
            return f'{cls}(\n{info}\n)'

    def stores_as(self, data: 'Graph'):
        return self

    @property
    def stores(self) -> List[BaseStorage]:
        return [self._store]

    @property
    def node_stores(self) -> List[NodeStorage]:
        return [self._store]

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        return [self._store]

    def to_dict(self) -> Dict[str, Any]:
        return self._store.to_dict()

    def to_namedtuple(self) -> NamedTuple:
        return self._store.to_namedtuple()

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        # if isinstance(value, SparseTensor) and 'adj' in key:
        # 	return (0, 1)
        if 'index' in key or key == 'face':
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(tlx.reduce_max(value)) + 1
        elif 'index' in key or key == 'face':
            return self.num_nodes
        else:
            return 0

    def debug(self):
        pass  # TODO

    def is_node_attr(self, key: str) -> bool:
        r"""Returns :obj:`True` if the object at :obj:`key` denotes a
        node-level attribute."""
        return self._store.is_node_attr(key)

    def is_edge_attr(self, key: str) -> bool:
        r"""Returns :obj:`True` if the object at key :obj:`key` denotes an
        edge-level attribute."""
        return self._store.is_edge_attr(key)

    #
    # def subgraph(self, subset: Tensor):
    # 	r"""Returns the induced subgraph given by the node indices
    # 	:obj:`subset`.
    # 	Args:
    # 		subset (LongTensor or BoolTensor): The nodes to keep.
    # 	"""
    #
    # 	out = subgraph(subset, self.edge_index, relabel_nodes=True,
    # 				   num_nodes=self.num_nodes, return_edge_mask=True)
    # 	edge_index, _, edge_mask = out
    #
    # 	if subset.dtype == torch.bool:
    # 		num_nodes = int(subset.sum())
    # 	else:
    # 		num_nodes = subset.size(0)
    #
    # 	data = copy.copy(self)
    #
    # 	for key, value in data:
    # 		if key == 'edge_index':
    # 			data.edge_index = edge_index
    # 		elif key == 'num_nodes':
    # 			data.num_nodes = num_nodes
    # 		elif isinstance(value, Tensor):
    # 			if self.is_node_attr(key):
    # 				data[key] = value[subset]
    # 			elif self.is_edge_attr(key):
    # 				data[key] = value[edge_mask]
    #
    # 	return data

    @property
    def in_degree(self):
        r"""
        Graph property, return the node in-degree of the graph.
        """
        if hasattr(self, 'csc_adj'):
            return self.csc_adj.degree
        return tlx.unsorted_segment_sum(tlx.ones(shape=(self.edge_index.shape[1],), dtype=tlx.int64),
                                        self.edge_index[1], self.num_nodes)

    @property
    def out_degree(self):
        r"""
        Graph property, return the node out-degree of the graph.
        """
        if hasattr(self, 'csr_adj'):
            return self.csr_adj.degree
        return tlx.unsorted_segment_sum(tlx.ones(shape=(self.edge_index.shape[1],), dtype=tlx.int64),
                                        self.edge_index[0], self.num_nodes)

    def add_self_loop(self, n_loops=1):
        """
        Parameters
        ----------
        n_loops: int

        Returns
        -------
        edge_index: Tensor
            original edges with self loop edges.
        edge_attr: FloatTensor
            attributes of edges.
        """
        return add_self_loops(self.edge_index, self.edge_attr, n_loops, num_nodes=self.num_nodes)

    def sorted_edges(self, sort_by="src"):
        """Return sorted edges with different strategies.
        This function will return sorted edges with different strategy.
        If :code:`sort_by="src"`, then edges will be sorted by :code:`src`
        nodes and otherwise :code:`dst`.

        Parameters
        ----------
        sort_by: str
            The type for sorted edges. ("src" or "dst")

        Returns
        -------
        tuple
            A tuple of (sorted_src, sorted_dst, sorted_eid).
        """
        if sort_by not in ["src", "dst"]:
            raise ValueError("sort_by should be in 'src' or 'dst'.")
        if sort_by == 'src':
            src, dst, eid = self._csr_adj.triples()
        else:
            dst, src, eid = self._csc_adj.triples()
        return src, dst, eid

    def tensor(self, inplace=True):
        """Convert the Graph into paddle.Tensor format.
        In paddle.Tensor format, the graph edges and node features are in paddle.Tensor format.
        You can use send and recv in paddle.Tensor graph.

        Parameters
        ----------
        inplace: bool
            (Default True) Whether to convert the graph into tensor inplace.

        """

        if inplace:
            for key in self._store:
                self._store[key] = self._apply_to_tensor(
                    key, self._store[key], inplace)
            return self
        else:
            new_dict = {}
            for key in self.__dict__:
                new_dict[key] = self._apply_to_tensor(key, self.__dict__[key],
                                                      inplace)

            graph = self.__class__(
                num_nodes=new_dict["_num_nodes"],
                edges=new_dict["_edges"],
                x=new_dict["x"],
                edge_attr=new_dict["edge_attr"],
                adj_src_index=new_dict["_adj_src_index"],
                adj_dst_index=new_dict["_adj_dst_index"],
                **new_dict)
            return graph

    def numpy(self, inplace=True):
        """Convert the Graph into numpy format.
        In numpy format, the graph edges and node features are in numpy.ndarray format.
        But you can't use send and recv in numpy graph.

        Parameters
        ----------
        inplace: bool
            (Default True) Whether to convert the graph into numpy inplace.

        """

        if inplace:

            for key in self._store:
                self._store[key] = self._apply_to_numpy(
                    key, self._store[key], inplace)
            return self
        else:
            new_dict = {}
            for key in self.__dict__:
                new_dict[key] = self._apply_to_numpy(key, self.__dict__[key],
                                                     inplace)

            graph = self.__class__(
                num_nodes=new_dict["_num_nodes"],
                edges=new_dict["_edges"],
                node_feat=new_dict["_node_feat"],
                edge_attr=new_dict["edge_attr"],
                adj_src_index=new_dict["_adj_src_index"],
                adj_dst_index=new_dict["_adj_dst_index"],
                **new_dict)
            return graph

    def to_heterogeneous(self, node_type=None,
                         edge_type=None,
                         node_type_names: Optional[List[NodeType]] = None,
                         edge_type_names: Optional[List[EdgeType]] = None):
        # (self, node_type: Optional[Tensor] = None,
        # edge_type: Optional[Tensor] = None,
        # node_type_names: Optional[List[NodeType]] = None,
        # edge_type_names: Optional[List[EdgeType]] = None):
        r"""Converts a :class:`~gammagl.data.Graph` object to a
        heterogeneous :class:`~gammagl.data.HeteroGraph` object.
        For this, node and edge attributes are splitted according to the
        node-level and edge-level vectors :obj:`node_type` and
        :obj:`edge_type`, respectively.
        :obj:`node_type_names` and :obj:`edge_type_names` can be used to give
        meaningful node and edge type names, respectively.
        That is, the node_type :obj:`0` is given by :obj:`node_type_names[0]`.
        If the :class:`~gammagl.data.Graph` object was constructed via
        :meth:`~gammagl.data.HeteroGraph.to_homogeneous`, the object can
        be reconstructed without any need to pass in additional arguments.

        Parameters
        ----------
            node_type: tensor, optional
                A node-level vector denoting the type
                of each node. (default: :obj:`None`)
            edge_type: tensor, optional
                An edge-level vector denoting the
                type of each edge. (default: :obj:`None`)
            node_type_names: list[str], optional
                The names of node types.
                (default: :obj:`None`)
            edge_type_names: list[tuple[str, str, str]], optional
                The names
                of edge types. (default: :obj:`None`)

        """
        from gammagl.data import HeteroGraph

        if node_type is None:
            node_type = self._store.get('node_type', None)
        if node_type is None:
            node_type = tlx.zeros(shape=(self.num_nodes,), dtype=tlx.int64)

        if node_type_names is None:
            store = self._store
            node_type_names = store.__dict__.get('_node_type_names', None)
        if node_type_names is None:
            node_type_names = [str(i) for i in np.unique(tlx.convert_to_numpy(node_type)).tolist()]

        if edge_type is None:
            edge_type = self._store.get('edge_type', None)
        if edge_type is None:
            edge_type = tlx.zeros(shape=(self.num_edges,), dtype=tlx.int64)

        if edge_type_names is None:
            store = self._store
            edge_type_names = store.__dict__.get('_edge_type_names', None)
        if edge_type_names is None:
            edge_type_names = []
            edge_index = self.edge_index
            for i in np.unique(tlx.convert_to_numpy(edge_type)).tolist():
                src, dst = tlx.mask_select(edge_index, edge_type == i, axis=1)
                src_types = np.unique(tlx.convert_to_numpy(tlx.gather(node_type, src))).tolist()
                dst_types = np.unique(tlx.convert_to_numpy(tlx.gather(node_type, dst))).tolist()
                if len(src_types) != 1 and len(dst_types) != 1:
                    raise ValueError(
                        "Could not construct a 'Graph' object from the "
                        "'Graph' object because single edge types span over "
                        "multiple node types")
                edge_type_names.append((node_type_names[src_types[0]], str(i),
                                        node_type_names[dst_types[0]]))

        # We iterate over node types to find the local node indices belonging
        # to each node type. Furthermore, we create a global `index_map` vector
        # that maps global node indices to local ones in the final
        # heterogeneous graph:
        node_ids, index_map = {}, tlx.zeros_like(node_type)
        for i, key in enumerate(node_type_names):
            idx = tlx.convert_to_numpy((node_type == i)).nonzero()[0]
            node_ids[i] = tlx.convert_to_tensor(idx)
            # node_ids[i] = (node_type == i).nonzero(as_tuple=False).view(-1)
            # index_map[node_ids[i]] = tlx.arange(start=0, limit=len(node_ids[i]))
            index_map = tlx.scatter_update(index_map, node_ids[i],
                                           tlx.arange(start=0, limit=len(node_ids[i]), dtype=tlx.int64))

        # We iterate over edge types to find the local edge indices:
        edge_ids = {}
        for i, key in enumerate(edge_type_names):
            idx = tlx.convert_to_numpy((edge_type == i)).nonzero()[0]
            edge_ids[i] = tlx.convert_to_tensor(idx)
        # edge_ids[i] = (edge_type == i).nonzero(as_tuple=False).view(-1)

        data = HeteroGraph()

        for i, key in enumerate(node_type_names):
            for attr, value in self.items():
                if attr == 'node_type' or attr == 'edge_type':
                    continue
                elif tlx.is_tensor(value) and self.is_node_attr(attr):
                    data[key][attr] = tlx.gather(value, node_ids[i])

            if len(data[key]) == 0:
                data[key].num_nodes = tlx.get_tensor_shape(node_ids[i])[0]

        for i, key in enumerate(edge_type_names):
            src, _, dst = key
            for attr, value in self.items():
                if attr == 'node_type' or attr == 'edge_type':
                    continue
                elif attr == 'edge_index':
                    edge_index = tlx.gather(value, edge_ids[i], axis=1)
                    edge_index = tlx.stack([tlx.gather(index_map, edge_index[0]), tlx.gather(index_map, edge_index[1])])
                    data[key].edge_index = edge_index
                elif tlx.is_tensor(value) and self.is_edge_attr(attr):
                    data[key][attr] = tlx.gather(value, edge_ids[i])

        # Add global attributes.
        keys = set(data.keys) | {'node_type', 'edge_type', 'num_nodes'}
        for attr, value in self.items():
            if attr in keys:
                continue
            if len(data.node_stores) == 1:
                data.node_stores[0][attr] = value
            else:
                data[attr] = value

        return data

    ###########################################################################

    @classmethod
    def from_dict(cls, mapping: Dict[str, Any]):
        r"""Creates a :class:`~gammagl.data.Graph` object from a Python
        dictionary."""
        return cls(**mapping)

    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the graph."""
        return self._store.num_node_features

    @property
    def num_features(self) -> int:
        r"""Returns the number of features per node in the graph.
        Alias for :py:attr:`~num_node_features`."""
        return self.num_node_features

    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per edge in the graph."""
        return self._store.num_edge_features

    # class `Graph` can't be collections.Iterable, or can't be saved in paddle.
    # def __iter__(self) -> Iterable:
    # 	r"""Iterates over all attributes in the data, yielding their attribute
    # 	names and values."""
    # 	for key, value in self._store.items():
    # 		yield key, value

    # do not use __iter__ function, or will throw exception in paddle backend.
    # So here use iter
    # now visit graph attr by for key, value in "graph.iter()", instead of "graph"
    def iter(self) -> Iterable:
        for key, value in self._store.items():
            yield key, value

    def __call__(self, *args: List[str]) -> Iterable:
        r"""Iterates over all attributes :obj:`*args` in the data, yielding
        their attribute names and values.
        If :obj:`*args` is not given, will iterate over all attributes."""
        for key, value in self._store.items(*args):
            yield key, value

    @property
    def x(self) -> Any:
        return self['x'] if 'x' in self._store else None

    @property
    def edge_index(self) -> Any:
        return self['edge_index'] if 'edge_index' in self._store else None

    @property
    def edge_weight(self) -> Any:
        return self['edge_weight'] if 'edge_weight' in self._store else None

    @property
    def edge_attr(self) -> Any:
        return self['edge_attr'] if 'edge_attr' in self._store else None

    @property
    def y(self) -> Any:
        return self['y'] if 'y' in self._store else None

    @property
    def pos(self) -> Any:
        return self['pos'] if 'pos' in self._store else None

    @property
    def batch(self) -> Any:
        return self['batch'] if 'batch' in self._store else None

    ###########################################################################

    def dump(self, path):
        r"""
        Dump the graph into a directory.

        This function will dump the graph information into the given directory path.
        The graph can be read back with :code:`pgl.Graph.load`

        Parameters
        ----------
        path: str
            The directory for the storage of the graph.
        """
        pass

    # if self._is_tensor:
    #     # Convert back into numpy and dump.
    #     graph = self.numpy(inplace=False)
    #     graph.dump(path)
    # else:
    #     if not os.path.exists(path):
    #         os.makedirs(path)

    #     np.save(os.path.join(path, 'num_nodes.npy'), self._num_nodes)
    #     np.save(os.path.join(path, 'edges.npy'), self._edges)
    #     np.save(os.path.join(path, 'num_graph.npy'), self._num_graph)

    #     if self._adj_src_index is not None:
    #         self._adj_src_index.dump(os.path.join(path, 'adj_src'))

    #     if self._adj_dst_index is not None:
    #         self._adj_dst_index.dump(os.path.join(path, 'adj_dst'))

    #     if self._graph_node_index is not None:
    #         np.save(
    #             os.path.join(path, 'graph_node_index.npy'),
    #             self._graph_node_index)

    #     if self._graph_edge_index is not None:
    #         np.save(
    #             os.path.join(path, 'graph_edge_index.npy'),
    #             self._graph_edge_index)

    #     def _dump_feat(feat_path, feat):
    #         """Dump all features to .npy file.
    #         """
    #         if len(feat) == 0:
    #             return

    #         if not os.path.exists(feat_path):
    #             os.makedirs(feat_path)

    #         for key in feat:
    #             value = feat[key]
    #             np.save(os.path.join(feat_path, key + ".npy"), value)

    #     _dump_feat(os.path.join(path, "node_feat"), self.node_feat)
    #     _dump_feat(os.path.join(path, "edge_feat"), self.edge_feat)

    @classmethod
    def load(cls, path, mmap_mode="r"):
        """Load Graph from path and return a Graph in numpy.

        Parameters
        ----------
        path: str
            The directory path of the stored Graph.
        mmap_mode: str
            Default :code:`mmap_mode="r"`. If not None, memory-map the graph.
        """
        pass

# num_nodes = np.load(
#     os.path.join(path, 'num_nodes.npy'), mmap_mode=mmap_mode)
# edges = np.load(os.path.join(path, 'edges.npy'), mmap_mode=mmap_mode)
# num_graph = np.load(
#     os.path.join(path, 'num_graph.npy'), mmap_mode=mmap_mode)
# if os.path.exists(os.path.join(path, 'graph_node_index.npy')):
#     graph_node_index = np.load(
#         os.path.join(path, 'graph_node_index.npy'),
#         mmap_mode=mmap_mode)
# else:
#     graph_node_index = None

# if os.path.exists(os.path.join(path, 'graph_edge_index.npy')):
#     graph_edge_index = np.load(
#         os.path.join(path, 'graph_edge_index.npy'),
#         mmap_mode=mmap_mode)
# else:
#     graph_edge_index = None

# if os.path.isdir(os.path.join(path, 'adj_src')):
#     adj_src_index = EdgeIndex.load(
#         os.path.join(path, 'adj_src'), mmap_mode=mmap_mode)
# else:
#     adj_src_index = None

# if os.path.isdir(os.path.join(path, 'adj_dst')):
#     adj_dst_index = EdgeIndex.load(
#         os.path.join(path, 'adj_dst'), mmap_mode=mmap_mode)
# else:
#     adj_dst_index = None

# def _load_feat(feat_path):
#     """Load features from .npy file.
#     """
#     feat = {}
#     if os.path.isdir(feat_path):
#         for feat_name in os.listdir(feat_path):
#             feat[os.path.splitext(feat_name)[0]] = np.load(
#                 os.path.join(feat_path, feat_name),
#                 mmap_mode=mmap_mode)
#     return feat

# node_feat = _load_feat(os.path.join(path, 'node_feat'))
# edge_feat = _load_feat(os.path.join(path, 'edge_feat'))
# return cls(edges=edges,
#            num_nodes=num_nodes,
#            node_feat=node_feat,
#            edge_feat=edge_feat,
#            adj_src_index=adj_src_index,
#            adj_dst_index=adj_dst_index,
#            _num_graph=num_graph,
#            _graph_node_index=graph_node_index,
#            _graph_edge_index=graph_edge_index)


def size_repr(key: Any, value: Any, indent: int = 0) -> str:
    pad = ' ' * indent
    if tlx.is_tensor(value) and value.shape == 0:
        out = value.item()
    elif tlx.is_tensor(value):
        out = str(list(value.shape))
    elif isinstance(value, np.ndarray):
        out = str(list(value.shape))
    # elif isinstance(value, SparseTensor):
    #     out = str(value.sizes())[:-1] + f', nnz={value.nnz()}]'
    elif isinstance(value, str):
        out = f"'{value}'"
    elif isinstance(value, Sequence):
        out = str([len(value)])
    elif isinstance(value, Mapping) and len(value) == 0:
        out = '{}'
    elif (isinstance(value, Mapping) and len(value) == 1
          and not isinstance(list(value.values())[0], Mapping)):
        lines = [size_repr(k, v, 0) for k, v in value.items()]
        out = '{ ' + ', '.join(lines) + ' }'
    elif isinstance(value, Mapping):
        lines = [size_repr(k, v, indent + 2) for k, v in value.items()]
        out = '{\n' + ',\n'.join(lines) + '\n' + pad + '}'
    else:
        out = str(value)

    key = str(key).replace("'", '')
    if isinstance(value, BaseStorage):
        return f'{pad}\033[1m{key}\033[0m={out}'
    else:
        return f'{pad}{key}={out}'
