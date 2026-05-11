import numpy as np
import networkx as nx
import tensorlayerx as tlx
from collections import OrderedDict, deque
from typing import Optional, Callable, Union


def _edge_index_to_numpy(edge_index):
    """Convert edge_index from any tensor type to numpy (2, E) int64."""
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
    return edge_np.astype(np.int64, copy=False)


def _add_self_loops(edge_index, num_nodes):
    """Add self-loops to edge_index. Returns numpy array."""
    ei = _edge_index_to_numpy(edge_index)
    num_nodes = int(num_nodes)
    self_loops = np.stack([np.arange(num_nodes), np.arange(num_nodes)], axis=0).astype(np.int64)
    if ei.shape[1] == 0:
        return self_loops
    result = np.concatenate([ei, self_loops], axis=1)
    return result


def _edge_index_to_nx(edge_index, num_nodes):
    """Convert edge_index to networkx Graph."""
    G = nx.Graph()
    G.add_nodes_from(range(int(num_nodes)))
    ei = _edge_index_to_numpy(edge_index)
    for i in range(ei.shape[1]):
        src, dst = int(ei[0, i]), int(ei[1, i])
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            G.add_edge(src, dst)
    return G


def _nx_to_edge_index(G):
    """Convert networkx Graph to numpy edge_index (2, E)."""
    edges = list(G.edges())
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(edges, dtype=np.int64).T


def _build_adj_list(edge_np, num_nodes, undirected=True):
    """Build adjacency list from edge_index numpy array."""
    adj = [[] for _ in range(int(num_nodes))]
    if edge_np.size == 0:
        return adj
    src = edge_np[0].astype(np.int64, copy=False)
    dst = edge_np[1].astype(np.int64, copy=False)
    for s, d in zip(src, dst):
        s, d = int(s), int(d)
        if 0 <= s < num_nodes and 0 <= d < num_nodes:
            adj[s].append(d)
            if undirected and s != d:
                adj[d].append(s)
    return adj


class _BatchGraph:
    """Lightweight container matching PyG Batch interface for tree/cycle/sequence."""
    __slots__ = ('edge_index', 'num_nodes', '_num_graphs')

    def __init__(self, edge_index, num_nodes, num_graphs):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self._num_graphs = num_graphs

    def __len__(self):
        return self._num_graphs


def _batch_from_graph_list(graph_list, num_nodes_per_graph):
    """Batch a list of (numpy edge_index) into a _BatchGraph.
    All graphs must have the same num_nodes_per_graph.
    """
    N = num_nodes_per_graph
    num_graphs = len(graph_list)
    total_nodes = num_graphs * N
    all_edges = []
    for i, ei in enumerate(graph_list):
        if ei.shape[1] > 0:
            all_edges.append(ei + i * N)
    if all_edges:
        edge_index = np.concatenate(all_edges, axis=1)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    return _BatchGraph(
        edge_index=tlx.convert_to_tensor(edge_index, dtype=tlx.int64),
        num_nodes=total_nodes,
        num_graphs=num_graphs,
    )


def _neighbor_sample(adj, seeds, num_neighbors):
    """Manual k-hop neighbor sampling.
    Returns (node_ids, local_edge_index, batch_size).
    """
    sampled = set(int(s) for s in seeds)
    frontier = list(sampled)
    hops = num_neighbors if isinstance(num_neighbors, (list, tuple)) else [num_neighbors]

    for hop_k in hops:
        next_frontier = []
        for node in frontier:
            neighs = adj[int(node)]
            if not neighs:
                continue
            if hop_k is None or int(hop_k) < 0 or int(hop_k) >= len(neighs):
                picked = neighs
            else:
                k = min(int(hop_k), len(neighs))
                picked = list(np.random.choice(neighs, size=k, replace=False))
            for nb in picked:
                nb = int(nb)
                if nb not in sampled:
                    sampled.add(nb)
                    next_frontier.append(nb)
        frontier = next_frontier

    node_list = sorted(sampled)
    node_map = {n: i for i, n in enumerate(node_list)}
    edges = []
    for u in node_list:
        for v in adj[u]:
            if v in node_map:
                edges.append((node_map[u], node_map[v]))
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)

    return np.array(node_list, dtype=np.int64), edge_index, len(seeds)


class _NodeSampler:
    """Manual neighbor-aware node sampler replacing PyG NeighborLoader."""

    def __init__(self, data, num_neighbors, batch_size, shuffle=True):
        edge_np = _edge_index_to_numpy(data.edge_index)
        num_nodes = int(data.num_nodes)
        self.adj = _build_adj_list(edge_np, num_nodes, undirected=True)
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_nodes = num_nodes

    def __len__(self):
        return max(1, int(np.ceil(self.num_nodes / self.batch_size)))

    def __iter__(self):
        indices = np.arange(self.num_nodes)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.num_nodes, self.batch_size):
            batch_seeds = indices[start:start + self.batch_size]
            n_id, edge_index, batch_size_val = _neighbor_sample(
                self.adj, batch_seeds, self.num_neighbors
            )

            class _Batch:
                pass
            result = _Batch()
            result.n_id = tlx.convert_to_tensor(n_id, dtype=tlx.int64)
            result.edge_index = tlx.convert_to_tensor(edge_index, dtype=tlx.int64)
            result.batch_size = batch_size_val
            result.num_nodes = len(n_id)
            result.batch = tlx.zeros((len(n_id),), dtype=tlx.int64)
            yield result


class _LinkSampler:
    """Manual edge-based neighbor sampler replacing PyG LinkNeighborLoader."""

    def __init__(self, data, num_neighbors, edge_label_index, batch_size, shuffle=True):
        edge_np = _edge_index_to_numpy(data.edge_index)
        num_nodes = int(data.num_nodes)
        self.adj = _build_adj_list(edge_np, num_nodes, undirected=True)
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle

        eli = _edge_index_to_numpy(edge_label_index)
        self.edges = [(int(eli[0, i]), int(eli[1, i])) for i in range(eli.shape[1])]
        self.num_edges = len(self.edges)
        self.num_nodes = num_nodes

    def __len__(self):
        return max(1, int(np.ceil(self.num_edges / self.batch_size)))

    def __iter__(self):
        indices = np.arange(self.num_edges)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.num_edges, self.batch_size):
            batch_edge_indices = indices[start:start + self.batch_size]
            batch_edges = [self.edges[i] for i in batch_edge_indices]

            seeds = set()
            for s, d in batch_edges:
                seeds.add(s)
                seeds.add(d)
            seeds = list(seeds)

            n_id, edge_index, _ = _neighbor_sample(self.adj, seeds, self.num_neighbors)

            local_edges = []
            id_to_local = {int(n): i for i, n in enumerate(n_id)}
            for s, d in batch_edges:
                if s in id_to_local and d in id_to_local:
                    local_edges.append((id_to_local[s], id_to_local[d]))
            edge_label_index_batch = np.array(local_edges, dtype=np.int64).T if local_edges else np.zeros((2, 0), dtype=np.int64)

            class _Batch:
                pass
            result = _Batch()
            result.n_id = tlx.convert_to_tensor(n_id, dtype=tlx.int64)
            result.edge_index = tlx.convert_to_tensor(edge_index, dtype=tlx.int64)
            result.edge_label_index = tlx.convert_to_tensor(edge_label_index_batch, dtype=tlx.int64)
            result.edge_label = tlx.convert_to_tensor(np.ones(len(local_edges), dtype=np.float32), dtype=tlx.float32)
            result.batch_size = len(batch_edges)
            result.num_nodes = len(n_id)
            yield result


class _GraphBatcher:
    """Simple graph batcher replacing PyG DataLoader for graph-level tasks."""

    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return max(1, int(np.ceil(len(self.dataset) / self.batch_size)))

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            graphs = [self.dataset[int(i)] for i in batch_indices]

            all_edges = []
            all_x = []
            all_y = []
            all_batch = []
            node_offset = 0
            for i, g in enumerate(graphs):
                ei = _edge_index_to_numpy(g.edge_index)
                if ei.shape[1] > 0:
                    all_edges.append(ei + node_offset)
                x_np = g.x.numpy() if hasattr(g.x, 'numpy') else np.array(g.x)
                all_x.append(x_np)
                if hasattr(g, 'y') and g.y is not None:
                    y_np = g.y.numpy() if hasattr(g.y, 'numpy') else np.array(g.y)
                    all_y.append(y_np)
                n = x_np.shape[0]
                all_batch.append(np.full(n, i, dtype=np.int64))
                node_offset += n

            edge_index = np.concatenate(all_edges, axis=1) if all_edges else np.zeros((2, 0), dtype=np.int64)
            x = np.concatenate(all_x, axis=0) if all_x else np.zeros((0, 0), dtype=np.float32)

            class _Batch:
                pass
            result = _Batch()
            result.edge_index = tlx.convert_to_tensor(edge_index, dtype=tlx.int64)
            result.x = tlx.convert_to_tensor(x, dtype=tlx.float32)
            result.batch = tlx.convert_to_tensor(np.concatenate(all_batch), dtype=tlx.int64) if all_batch else tlx.zeros((0,), dtype=tlx.int64)
            result.num_nodes = node_offset
            result.y = tlx.convert_to_tensor(np.concatenate(all_y), dtype=tlx.int64) if all_y else None
            yield result


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def __contains__(self, item):
        return item in self.cache

    def clear(self):
        self.cache.clear()


def _sample_sequence(edge_index, start_node, sequence_length=5, adj=None):
    """Sample a sequence of nodes starting from start_node via BFS."""
    sequence = [int(start_node)]
    queue = deque([int(start_node)])

    if adj is None:
        ei = _edge_index_to_numpy(edge_index)
        src_nodes = ei[0]
        dst_nodes = ei[1]
    else:
        src_nodes = dst_nodes = None

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


def _build_tree_cycle_sequence(G, n_id_list, batch_size, max_cycle_edges, max_seq_edges,
                               sample_sequence_fn, adj_for_seq=None):
    """Build tree, cycle, and sequence subgraphs for one batch.
    Returns (tree_edges, cycle_edges, sequence_edges) as lists of numpy edge_index arrays.
    """
    num_nodes_per_graph = len(n_id_list)
    tree_edges_list = []
    cycle_edges_list = []
    seq_edges_list = []

    for m in range(batch_size):
        # Tree: full BFS tree from seed node m
        try:
            tree_edges = sorted(list(nx.bfs_tree(G, m).edges()))
        except Exception:
            tree_edges = []
        tG = nx.Graph()
        tG.add_edges_from(tree_edges)
        tree_ei = _nx_to_edge_index(tG)
        tree_edges_list.append(tree_ei)

        # Cycle: first few BFS edges
        bfs_edges = sorted(list(nx.bfs_edges(G, m)))
        cycle_limit = max_cycle_edges - 1
        max_edges = bfs_edges[:cycle_limit]
        cG = nx.Graph()
        cG.add_edges_from(max_edges)
        cycle_nodes = list(cG.nodes)
        if len(cycle_nodes) == max_cycle_edges and cycle_nodes[0] == cycle_nodes[-1]:
            cycle_ei = _nx_to_edge_index(cG)
        else:
            seq = sample_sequence_fn(m, sequence_length=max_cycle_edges)
            seq = [int(s) for s in seq]
            cG2 = nx.Graph([(seq[i], seq[i + 1]) for i in range(len(seq) - 1)])
            cycle_ei = _nx_to_edge_index(cG2)
        cycle_edges_list.append(cycle_ei)

        # Sequence: BFS edges truncated
        seq_edges = sorted(list(nx.bfs_edges(G, m)))
        seq_limit = max_seq_edges - 1
        max_edges_seq = seq_edges[:seq_limit]
        sG = nx.Graph()
        sG.add_edges_from(max_edges_seq)
        seq_ei = _nx_to_edge_index(sG)
        seq_edges_list.append(seq_ei)

    return tree_edges_list, cycle_edges_list, seq_edges_list


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

        self._sampler = _NodeSampler(data, num_neighbors, batch_size=batch_size, shuffle=shuffle)

        edge_np = _edge_index_to_numpy(data.edge_index)
        self._data_adj = _build_adj_list(edge_np, int(data.num_nodes), undirected=True)

        self._max_cycle_edges = max_depth_cycle
        self._max_seq_edges = sequence_length

    def __len__(self):
        return len(self._sampler)

    def __iter__(self):
        for key, batch_data in enumerate(self._sampler):
            batch_data.edge_index = tlx.convert_to_tensor(
                _add_self_loops(batch_data.edge_index, batch_data.num_nodes),
                dtype=tlx.int64
            )

            cached = self.cache.get(key)
            if cached is not None:
                yield cached
                continue

            subset = batch_data.n_id
            sub_edge_index = batch_data.edge_index
            num_sub_nodes = int(batch_data.num_nodes)

            G = _edge_index_to_nx(sub_edge_index, num_sub_nodes)

            n_id_list = [int(x) for x in (subset.tolist() if hasattr(subset, 'tolist')
                                          else list(tlx.convert_to_numpy(subset)))]
            bs = int(batch_data.batch_size)

            tree_edges, cycle_edges, seq_edges = _build_tree_cycle_sequence(
                G, n_id_list, bs,
                max_cycle_edges=self._max_cycle_edges,
                max_seq_edges=self._max_seq_edges,
                sample_sequence_fn=lambda start, seq_len: _sample_sequence(
                    sub_edge_index, start, seq_len, adj=None
                ),
            )

            batch_tree = _batch_from_graph_list(tree_edges, num_sub_nodes)
            batch_cycle = _batch_from_graph_list(cycle_edges, num_sub_nodes)
            batch_sequence = _batch_from_graph_list(seq_edges, num_sub_nodes)

            batch_data.batch_tree = batch_tree
            batch_data.batch_cycle = batch_cycle
            batch_data.batch_sequence = batch_sequence

            self.cache.put(key, batch_data)
            yield batch_data

    def clear_cache(self):
        self.cache.clear()


class ExtractLinkLoader:
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
                 capacity: int = 1000,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 **kwargs):

        self.data = data
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.cache = LRUCache(capacity=capacity)
        self.shuffle = shuffle

        self._sampler = _LinkSampler(data, num_neighbors, edge_label_index, batch_size, shuffle)

        self._max_cycle_edges = 3
        self._max_seq_edges = 5

    def __len__(self):
        return len(self._sampler)

    def __iter__(self):
        for key, batch_data in enumerate(self._sampler):
            batch_data.edge_index = tlx.convert_to_tensor(
                _add_self_loops(batch_data.edge_index, batch_data.num_nodes),
                dtype=tlx.int64
            )

            cached = self.cache.get(key)
            if cached is not None:
                yield cached
                continue

            subset = batch_data.n_id
            sub_edge_index = batch_data.edge_index
            num_sub_nodes = int(batch_data.num_nodes)

            G = _edge_index_to_nx(sub_edge_index, num_sub_nodes)

            n_id_list = [int(x) for x in (subset.tolist() if hasattr(subset, 'tolist')
                                          else list(tlx.convert_to_numpy(subset)))]
            bs = int(batch_data.batch_size)

            tree_edges, cycle_edges, seq_edges = _build_tree_cycle_sequence(
                G, n_id_list, bs,
                max_cycle_edges=self._max_cycle_edges,
                max_seq_edges=self._max_seq_edges,
                sample_sequence_fn=lambda start, seq_len: _sample_sequence(
                    sub_edge_index, start, seq_len, adj=None
                ),
            )

            batch_tree = _batch_from_graph_list(tree_edges, num_sub_nodes)
            batch_cycle = _batch_from_graph_list(cycle_edges, num_sub_nodes)
            batch_sequence = _batch_from_graph_list(seq_edges, num_sub_nodes)

            batch_data.batch_tree = batch_tree
            batch_data.batch_cycle = batch_cycle
            batch_data.batch_sequence = batch_sequence

            self.cache.put(key, batch_data)
            yield batch_data

    def clear_cache(self):
        self.cache.clear()


class ExtractGraphLoader:
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
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache = LRUCache(capacity=capacity)
        self.cn = centroid_num

        self._batcher = _GraphBatcher(dataset, batch_size, shuffle)

    def __len__(self):
        return len(self._batcher)

    def __iter__(self):
        for key, batch_data in enumerate(self._batcher):
            batch_data.edge_index = tlx.convert_to_tensor(
                _add_self_loops(batch_data.edge_index, batch_data.num_nodes),
                dtype=tlx.int64
            )

            cached = self.cache.get(key)
            if cached is not None:
                yield cached
                continue

            tree_edges_list = []
            num_sub_nodes = int(batch_data.num_nodes)

            ptr = tlx.ones((num_sub_nodes,), dtype=tlx.float32)
            num_per_G = tlx.convert_to_numpy(
                tlx.cast(tlx.reduce_sum(ptr, axis=0, keepdims=False),
                         tlx.int64)
            )
            # For batched graph classification, compute graph sizes from the batch attribute
            batch_arr = tlx.convert_to_numpy(batch_data.batch) if hasattr(batch_data, 'batch') else np.zeros(num_sub_nodes, dtype=np.int64)
            num_graphs = int(batch_arr.max()) + 1 if len(batch_arr) > 0 else 1
            counts = np.bincount(batch_arr, minlength=num_graphs)
            if len(counts) == 1 and counts[0] != num_sub_nodes:
                num_per_G = np.array([num_sub_nodes], dtype=np.int64)
            else:
                num_per_G = counts

            n_id = np.arange(num_sub_nodes)
            subset_list = []
            cur = 0
            for cnt in num_per_G:
                cnt = int(cnt)
                end = cur + cnt
                k = min(self.cn, cnt)
                subset_list.append(np.random.choice(n_id[cur:end], k, replace=False))
                cur = end
            subset = np.concatenate(subset_list, axis=0) if subset_list else np.array([], dtype=np.int64)

            G = _edge_index_to_nx(batch_data.edge_index, num_sub_nodes)
            for seed_node in subset:
                seed = int(seed_node)
                try:
                    tree_edges = sorted(list(nx.bfs_tree(G, seed).edges()))
                except Exception:
                    tree_edges = []
                tG = nx.Graph()
                tG.add_edges_from(tree_edges)
                tree_ei = _nx_to_edge_index(tG)
                tree_edges_list.append(tree_ei)

            batch_tree = _batch_from_graph_list(tree_edges_list, num_sub_nodes)
            batch_data.batch_tree = batch_tree

            self.cache.put(key, batch_data)
            yield batch_data

    def clear_cache(self):
        self.cache.clear()
