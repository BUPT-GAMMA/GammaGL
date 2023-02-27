import tensorlayerx as tlx
import numpy as np
from itertools import chain

from scipy.sparse.csgraph import shortest_path

from gammagl.utils import k_hop_subgraph, to_scipy_sparse_matrix
from gammagl.data import Graph, InMemoryDataset
from gammagl.transforms import RandomLinkSplit

class SEALDataset(InMemoryDataset):
    def __init__(self, dataset, env, num_hops, neg_sampling_ratio=1.0, split='train'):
        self.data = dataset[0]
        self.num_hops = num_hops
        self.neg_sampling_ratio = neg_sampling_ratio
        self.env = env
        super().__init__(dataset.root)
        index = ['train', 'val', 'test'].index(split)
        self.data, self.slices = tlx.files.load_npy_to_any(name=self.processed_paths[index])

    @property
    def processed_file_names(self):
        return [f'{self.env}_SEAL_train_data.npy', f'{self.env}_SEAL_val_data.npy', f'{self.env}_SEAL_test_data.npy']

    def process(self):
        transform = RandomLinkSplit(num_val=0.05, num_test=0.1, neg_sampling_ratio=self.neg_sampling_ratio,
                                    is_undirected=True, split_labels=True)
        train_data, val_data, test_data = transform(self.data)

        self._max_z = 0

        # Collect a list of subgraphs for training, validation and testing:
        train_pos_data_list = self.extract_enclosing_subgraphs(
            train_data.edge_index, train_data.pos_edge_label_index, 1.)
        train_neg_data_list = self.extract_enclosing_subgraphs(
            train_data.edge_index, train_data.neg_edge_label_index, 0.)

        val_pos_data_list = self.extract_enclosing_subgraphs(
            val_data.edge_index, val_data.pos_edge_label_index, 1.)
        val_neg_data_list = self.extract_enclosing_subgraphs(
            val_data.edge_index, val_data.neg_edge_label_index, 0.)

        test_pos_data_list = self.extract_enclosing_subgraphs(
            test_data.edge_index, test_data.pos_edge_label_index, 1.)
        test_neg_data_list = self.extract_enclosing_subgraphs(
            test_data.edge_index, test_data.neg_edge_label_index, 0.)

        # Convert node labeling to one-hot features.
        for data in chain(train_pos_data_list, train_neg_data_list,
                          val_pos_data_list, val_neg_data_list,
                          test_pos_data_list, test_neg_data_list):
            data.x = tlx.reshape(
                tlx.scatter_update(
                    tlx.zeros((data.z.shape[0] * (self._max_z + 1),), dtype=tlx.float32),
                    data.z + tlx.arange(0, data.z.shape[0], dtype=data.z.dtype) * (self._max_z + 1),
                    tlx.ones((data.z.shape[0],), dtype=tlx.float32)
                ),
                (data.z.shape[0], self._max_z + 1)
            )

        tlx.files.save_any_to_npy(self.collate(train_pos_data_list + train_neg_data_list),
                   self.processed_paths[0])
        tlx.files.save_any_to_npy(self.collate(val_pos_data_list + val_neg_data_list),
                   self.processed_paths[1])
        tlx.files.save_any_to_npy(self.collate(test_pos_data_list + test_neg_data_list),
                   self.processed_paths[2])

    def extract_enclosing_subgraphs(self, edge_index, edge_label_index, y):
        data_list = []
        for src, dst in tlx.convert_to_numpy(edge_label_index).T.tolist():
            sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
                [src, dst], self.num_hops, edge_index, relabel_nodes=True)
            src, dst = mapping.tolist()

            # Remove target link from the subgraph.
            mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
            mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
            sub_edge_index = tlx.mask_select(sub_edge_index, mask1 & mask2, axis=1)

            # Calculate node labeling.
            z = self.drnl_node_labeling(sub_edge_index, src, dst, num_nodes=sub_nodes.shape[0])

            data = Graph(x=tlx.gather(self.data.x, sub_nodes), z=z, edge_index=sub_edge_index, y=y)
            data_list.append(data)

        return data_list

    def drnl_node_labeling(self, edge_index, src, dst, num_nodes=None):
        # Double-radius node labeling (DRNL).
        src, dst = (dst, src) if src > dst else (src, dst)
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

        idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src = adj[idx, :][:, idx]

        idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst = adj[idx, :][:, idx]

        dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True,
                                 indices=src)
        dist2src = np.insert(dist2src, dst, 0, axis=0)

        dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True,
                                 indices=dst - 1)
        dist2dst = np.insert(dist2dst, src, 0, axis=0)

        dist = dist2src + dist2dst
        dist_over_2, dist_mod_2 = dist // 2, dist % 2

        z = 1 + np.minimum(dist2src, dist2dst)
        z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
        z[src] = 1.
        z[dst] = 1.
        z = np.nan_to_num(z, nan=0.)
        z = tlx.convert_to_tensor(z, dtype=tlx.int64)

        self._max_z = max(tlx.reduce_max(z), self._max_z)

        return z