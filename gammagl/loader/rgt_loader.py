from torch_geometric.data import Data, Batch
import sys
import importlib
import torch

_HAS_PYG_LIB = False
_HAS_TORCH_SPARSE = False

try:
    importlib.import_module('pyg_lib')
    _HAS_PYG_LIB = True
except Exception:
    pass

try:
    importlib.import_module('torch_sparse')
    _HAS_TORCH_SPARSE = True
except Exception:
    pass

_USE_NEIGHBOR_SAMPLING = _HAS_PYG_LIB or _HAS_TORCH_SPARSE

if _USE_NEIGHBOR_SAMPLING:
    from torch_geometric.loader import NeighborLoader, LinkNeighborLoader, DataLoader
else:
    from torch_geometric.loader import DataLoader
    class NeighborLoader:
        pass
    class LinkNeighborLoader:
        pass
from typing import Union, List, Optional, Callable
from torch_geometric.utils import add_self_loops, to_networkx, from_networkx
from collections import OrderedDict
import networkx as nx
import numpy as np
from collections import deque
import tensorlayerx as tlx


def _to_pyg_data(data):
    if isinstance(data, Data):
        return data
    pyg_data = Data()
    edge_index = data.edge_index
    if isinstance(edge_index, torch.Tensor) and not edge_index.is_contiguous():
        edge_index = edge_index.contiguous()
    pyg_data.edge_index = edge_index
    if hasattr(data, 'x') and data.x is not None:
        x = data.x
        if isinstance(x, torch.Tensor) and not x.is_contiguous():
            x = x.contiguous()
        pyg_data.x = x
    if hasattr(data, 'y') and data.y is not None:
        y = data.y
        if isinstance(y, torch.Tensor) and not y.is_contiguous():
            y = y.contiguous()
        pyg_data.y = y
    if hasattr(data, 'tokens') and data.tokens is not None:
        pyg_data.tokens = data.tokens
    if hasattr(data, 'train_mask') and data.train_mask is not None:
        pyg_data.train_mask = data.train_mask
    if hasattr(data, 'val_mask') and data.val_mask is not None:
        pyg_data.val_mask = data.val_mask
    if hasattr(data, 'test_mask') and data.test_mask is not None:
        pyg_data.test_mask = data.test_mask
    if hasattr(data, 'num_nodes'):
        pyg_data.num_nodes = data.num_nodes
    return pyg_data


def _edge_index_to_numpy(edge_index):
    if hasattr(edge_index, "detach"):
        edge_np = edge_index.detach().cpu().numpy()
    elif hasattr(edge_index, "numpy"):
        edge_np = edge_index.numpy()
    else:
        edge_np = np.array(edge_index)
    edge_np = np.asarray(edge_np)
    if edge_np.ndim != 2:
        edge_np = edge_np.reshape(2, -1)
    if edge_np.shape[0] != 2 and edge_np.shape[1] == 2:
        edge_np = edge_np.T
    return edge_np


def _build_adj_list(edge_np, num_nodes, undirected=False):
    adj = [[] for _ in range(int(num_nodes))]
    if edge_np.size == 0:
        return adj
    src = edge_np[0].astype(np.int64, copy=False)
    dst = edge_np[1].astype(np.int64, copy=False)
    for s, d in zip(src, dst):
        if 0 <= s < num_nodes and 0 <= d < num_nodes:
            adj[s].append(d)
            if undirected and s != d:
                adj[d].append(s)
    return adj


def _bfs_edges(adj, start, max_edges=None):
    start = int(start)
    if start < 0 or start >= len(adj):
        return []
    visited = {start}
    queue = deque([start])
    edges = []
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            v = int(v)
            if v in visited:
                continue
            visited.add(v)
            queue.append(v)
            edges.append((u, v))
            if max_edges is not None and len(edges) >= int(max_edges):
                return edges
    return edges


def _edge_list_to_tensor(edge_list):
    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index_np = np.asarray(edge_list, dtype=np.int64).T
    return torch.tensor(edge_index_np, dtype=torch.long)


def _seed_cache_key(data, fallback_key):
    if hasattr(data, "n_id") and hasattr(data, "batch_size"):
        try:
            bs = int(data.batch_size)
            seeds = data.n_id[:bs]
            if hasattr(seeds, "detach"):
                seeds = seeds.detach().cpu().numpy()
            elif hasattr(seeds, "numpy"):
                seeds = seeds.numpy()
            else:
                seeds = np.asarray(seeds)
            seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
            return ("seed", tuple(seeds.tolist()))
        except Exception:
            pass
    return ("idx", int(fallback_key))


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def __contains__(self, item):
        return item in self.cache

    def clear(self):
        self.cache.clear()


class ExtractNodeLoader:
    def __init__(self, data,
                 num_neighbors,
                 input_nodes=None,
                 input_time=None,
                 replace: bool = False,
                 subgraph_type='directional',
                 disjoint: bool = False,
                 temporal_strategy: str = 'uniform',
                 time_attr=None,
                 weight_attr=None,
                 transform=None,
                 transform_sampler_output=None,
                 is_sorted: bool = False,
                 filter_per_worker: bool = False,
                 neighbor_sampler=None,
                 capacity: int = 1000,
                 max_cycle_samples: int = 3,
                 max_sequence_samples: int = 3,
                 max_depth_cycle: int = 3,
                 sequence_length: int = 4,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 **kwargs):

        self.data = data
        self.num_neighbors = num_neighbors
        self.input_nodes = input_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache = LRUCache(capacity=capacity)
        self.max_cycle_samples = max_cycle_samples
        self.max_sequence_samples = max_sequence_samples
        self.max_depth_cycle = max_depth_cycle
        self.sequence_length = sequence_length
        self._use_native_sampling = _USE_NEIGHBOR_SAMPLING

        if not self._use_native_sampling:
            raise ImportError(
                "ExtractNodeLoader requires native NeighborLoader backend (pyg_lib or torch_sparse) "
                "to match RGT-origin behavior. Install one of them before training."
            )

        pyg_data = _to_pyg_data(data)
        self._loader = NeighborLoader(
            pyg_data, num_neighbors, input_nodes, input_time, replace, subgraph_type,
            disjoint, temporal_strategy, time_attr, weight_attr, transform,
            transform_sampler_output, is_sorted, filter_per_worker, neighbor_sampler,
            batch_size=batch_size, shuffle=shuffle, **kwargs
        )

    def _expand_with_neighbors(self, seed_nodes):
        seeds = [int(n) for n in seed_nodes]
        seen = set(seeds)
        expanded = list(seeds)
        frontier = list(seeds)
        hops = self.num_neighbors if isinstance(self.num_neighbors, (list, tuple)) else [self.num_neighbors]

        for hop_k in hops:
            if len(frontier) == 0:
                break
            next_frontier = []
            for node in frontier:
                neigh = self._adj_undirected[int(node)] if 0 <= int(node) < len(self._adj_undirected) else []
                if len(neigh) == 0:
                    continue
                if hop_k is None or int(hop_k) < 0 or int(hop_k) >= len(neigh):
                    picked = neigh
                else:
                    k = int(hop_k)
                    idx = np.random.choice(len(neigh), size=k, replace=False)
                    picked = [neigh[i] for i in idx]
                for nb in picked:
                    nb = int(nb)
                    if nb not in seen:
                        seen.add(nb)
                        expanded.append(nb)
                        next_frontier.append(nb)
            frontier = next_frontier
        return expanded

    def sample_sequence(self, start_node, sequence_length=5, adj=None):
        sequence = [start_node]
        queue = deque([start_node])
        if adj is None:
            edge_index = self.data.edge_index
            if hasattr(edge_index, "detach"):
                edge_index = edge_index.detach().cpu().numpy()
            elif hasattr(edge_index, "numpy"):
                edge_index = edge_index.numpy()
            edge_index = np.asarray(edge_index)
            if edge_index.ndim != 2:
                edge_index = edge_index.reshape(2, -1)
            if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
                edge_index = edge_index.T
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]

        while len(sequence) < sequence_length and queue:
            node = queue.popleft()
            if adj is None:
                neighbors = dst_nodes[src_nodes == int(node)]
            else:
                neighbors = adj[int(node)] if 0 <= int(node) < len(adj) else []

            for neighbor in neighbors:
                nb = int(neighbor)
                if nb not in sequence:
                    queue.append(nb)
                    sequence.append(nb)

        return sequence

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        for key, data in enumerate(self._loader):
            data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
            if key in self.cache:
                data = self.cache.get(key)
            else:
                tree_list = []
                cycle_list = []
                sequence_list = []
                subset = data.n_id
                sub_edge_index = data.edge_index
                G = to_networkx(Data(edge_index=sub_edge_index, num_nodes=subset.shape[0]))
                n_id_list = subset.tolist() if hasattr(subset, 'tolist') else list(subset)
                for m, seed_node in enumerate(n_id_list[:data.batch_size]):
                    sorted_edges = sorted(list(nx.bfs_tree(G, m).edges()))
                    tG = nx.Graph()
                    tG.add_edges_from(sorted_edges)
                    tree_edge_index = from_networkx(tG).edge_index
                    del tG

                    sorted_edges = sorted(list(nx.bfs_edges(G, m)))
                    tG = nx.Graph()
                    max_edges = sorted_edges[:3 - 1]
                    tG.add_edges_from(max_edges)

                    cycle_nodes = list(tG.nodes)
                    if len(cycle_nodes) == 3 and cycle_nodes[0] == cycle_nodes[-1]:
                        cycle_edge_index = from_networkx(tG).edge_index
                    else:
                        sequence = self.sample_sequence(m, sequence_length=3)
                        sequence = [int(s) for s in sequence]
                        cycle_edge_index = from_networkx(
                            nx.Graph([(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)])
                        ).edge_index

                    del tG

                    sorted_edges_sequence = sorted(list(nx.bfs_edges(G, m)))
                    tG = nx.Graph()
                    max_edges = sorted_edges_sequence[:5 - 1]
                    tG.add_edges_from(max_edges)
                    sequence_edge_index = from_networkx(tG).edge_index
                    del tG

                    tree_list.append(Data(edge_index=tree_edge_index, num_nodes=subset.shape[0], seed_node=seed_node))
                    cycle_list.append(Data(edge_index=cycle_edge_index, num_nodes=subset.shape[0], seed_node=seed_node))
                    sequence_list.append(Data(edge_index=sequence_edge_index, num_nodes=subset.shape[0], seed_node=seed_node))
                batch_tree = Batch.from_data_list(tree_list)
                batch_cycle = Batch.from_data_list(cycle_list)
                batch_sequence = Batch.from_data_list(sequence_list)
                data.batch_tree = batch_tree
                data.batch_cycle = batch_cycle
                data.batch_sequence = batch_sequence

                self.cache.put(key, data)

            yield data

    def clear_cache(self):
        self.cache.clear()


class ExtractLinkLoader(LinkNeighborLoader):
    def __init__(self, data,
                 num_neighbors,
                 edge_label_index=None,
                 edge_label=None,
                 edge_label_time=None,
                 replace: bool = False,
                 subgraph_type='directional',
                 disjoint: bool = False,
                 temporal_strategy: str = 'uniform',
                 neg_sampling=None,
                 neg_sampling_ratio: Optional[Union[int, float]] = None,
                 time_attr: Optional[str] = None,
                 weight_attr=None,
                 transform: Optional[Callable] = None,
                 transform_sampler_output: Optional[Callable] = None,
                 is_sorted: bool = False,
                 filter_per_worker: bool = False,
                 neighbor_sampler=None,
                 capacity: int = 1000, **kwargs):
        pyg_data = _to_pyg_data(data)
        super(ExtractLinkLoader, self).__init__(
            pyg_data, num_neighbors, edge_label_index, edge_label, edge_label_time, replace, subgraph_type, disjoint,
            temporal_strategy,
            neg_sampling, neg_sampling_ratio, time_attr, weight_attr, transform, transform_sampler_output, is_sorted,
            filter_per_worker, neighbor_sampler, **kwargs
        )
        if not _USE_NEIGHBOR_SAMPLING:
            raise ImportError(
                "ExtractLinkLoader requires native LinkNeighborLoader backend (pyg_lib or torch_sparse) "
                "to match RGT-origin behavior. Install one of them before training."
            )
        self.cache = LRUCache(capacity=capacity)

    def sample_sequence(self, start_node, sequence_length=5, adj=None):
        sequence = [start_node]
        queue = deque([start_node])
        if adj is None:
            edge_index = self.data.edge_index
            if hasattr(edge_index, "detach"):
                edge_index = edge_index.detach().cpu().numpy()
            elif hasattr(edge_index, "numpy"):
                edge_index = edge_index.numpy()
            edge_index = np.asarray(edge_index)
            if edge_index.ndim != 2:
                edge_index = edge_index.reshape(2, -1)
            if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
                edge_index = edge_index.T
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]

        while len(sequence) < sequence_length and queue:
            node = queue.popleft()
            if adj is None:
                neighbors = dst_nodes[src_nodes == int(node)]
            else:
                neighbors = adj[int(node)] if 0 <= int(node) < len(adj) else []

            for neighbor in neighbors:
                nb = int(neighbor)
                if nb not in sequence:
                    queue.append(nb)
                    sequence.append(nb)

        return sequence

    def __iter__(self):
        for key, data in enumerate(super().__iter__()):
            data.batch_size = self.batch_size
            data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
            if key in self.cache:
                data = self.cache.get(key)
            else:
                tree_list = []
                cycle_list = []
                sequence_list = []
                subset = data.n_id
                sub_edge_index = data.edge_index
                G = to_networkx(Data(edge_index=sub_edge_index, num_nodes=subset.shape[0]))
                n_id_list = subset.tolist() if hasattr(subset, 'tolist') else list(subset)
                for m, seed_node in enumerate(n_id_list[:data.batch_size]):
                    sorted_edges = sorted(list(nx.bfs_tree(G, m).edges()))
                    tG = nx.Graph()
                    tG.add_edges_from(sorted_edges)
                    tree_edge_index = from_networkx(tG).edge_index
                    del tG

                    sorted_edges = sorted(list(nx.bfs_edges(G, m)))
                    tG = nx.Graph()
                    max_edges = sorted_edges[:3 - 1]
                    tG.add_edges_from(max_edges)

                    cycle_nodes = list(tG.nodes)
                    if len(cycle_nodes) == 3 and cycle_nodes[0] == cycle_nodes[-1]:
                        cycle_edge_index = from_networkx(tG).edge_index
                    else:
                        sequence = self.sample_sequence(m, sequence_length=3)
                        cycle_edge_index = from_networkx(
                            nx.Graph([(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)])
                        ).edge_index

                    del tG

                    sorted_edges_sequence = sorted(list(nx.bfs_edges(G, m)))
                    tG = nx.Graph()
                    max_edges = sorted_edges_sequence[:5 - 1]
                    tG.add_edges_from(max_edges)
                    sequence_edge_index = from_networkx(tG).edge_index
                    del tG

                    tree_list.append(Data(edge_index=tree_edge_index, num_nodes=subset.shape[0], seed_node=seed_node))
                    cycle_list.append(Data(edge_index=cycle_edge_index, num_nodes=subset.shape[0], seed_node=seed_node))
                    sequence_list.append(Data(edge_index=sequence_edge_index, num_nodes=subset.shape[0], seed_node=seed_node))
                batch_tree = Batch.from_data_list(tree_list)
                batch_cycle = Batch.from_data_list(cycle_list)
                batch_sequence = Batch.from_data_list(sequence_list)
                data.batch_tree = batch_tree
                data.batch_cycle = batch_cycle
                data.batch_sequence = batch_sequence

                self.cache.put(key, data)

            yield data

    def clear_cache(self):
        self.cache.clear()


class ExtractGraphLoader(DataLoader):
    def __init__(self,
                 dataset,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 follow_batch=None,
                 exclude_keys=None,
                 capacity: int = 1000,
                 centroid_num: int = 10,
                 **kwargs,
                 ):
        super(ExtractGraphLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            follow_batch,
            exclude_keys,
            **kwargs,
        )
        self.cache = LRUCache(capacity=capacity)
        self.cn = centroid_num

    def __iter__(self):
        for key, data in enumerate(super().__iter__()):
            data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
            if key in self.cache:
                data = self.cache.get(key)
            else:
                tree_list = []
                ptr = tlx.ones(data.num_nodes)
                num_per_G = tlx.cast(tlx.reduce_sum(ptr, dim=data.batch), tlx.int64).numpy()
                n_id = np.arange(data.num_nodes)
                subset = []
                cur = 0
                for i in num_per_G:
                    subset.append(np.random.choice(n_id[cur: cur + i], self.cn, replace=False))
                    cur += i
                subset = np.concatenate(subset, axis=0)
                G = to_networkx(Data(edge_index=data.edge_index, num_nodes=data.num_nodes))
                for m, seed_node in enumerate(subset):
                    sorted_edges = sorted(list(nx.bfs_tree(G, seed_node).edges()))
                    tG = nx.Graph()
                    tG.add_edges_from(sorted_edges)
                    tree_edge_index = from_networkx(tG).edge_index
                    del tG
                    tree_list.append(Data(edge_index=tree_edge_index, num_nodes=subset.shape[0], seed_node=seed_node))
                batch_tree = Batch.from_data_list(tree_list)
                data.batch_tree = batch_tree
                self.cache.put(key, data)
            yield data

    def clear_cache(self):
        self.cache.clear()
