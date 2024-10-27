import os.path as osp
import numpy as np
import tensorlayerx as tlx
from gammagl.data import InMemoryDataset, HeteroGraph, download_url
from scipy import io as sio
from typing import Callable, List, Optional
import scipy.sparse as sp

class ACM4Rohe(InMemoryDataset):
    url = "https://github.com/Jhy1993/HAN/raw/master/data/acm/ACM.mat"

    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, force_reload: bool = False):
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["ACM.mat"]

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + "_data.pt"

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        # Load the original ACM data
        data_path = osp.join(self.raw_dir, "ACM.mat")
        data = sio.loadmat(data_path)

        # Parse data, including matrices for paper-author, paper-field relationships
        p_vs_f = data["PvsL"]  # Paper-field matrix
        p_vs_a = data["PvsA"]  # Paper-author matrix
        p_vs_t = data["PvsT"]  # Paper feature matrix
        p_vs_c = data["PvsC"]  # Paper-conference labels matrix

        # Assign classes to specific conferences
        conf_ids = [0, 1, 9, 10, 13]
        label_ids = [0, 1, 2, 2, 1]

        # Filter papers with specific conference labels
        p_vs_c_filter = p_vs_c[:, conf_ids]
        p_selected = np.nonzero(p_vs_c_filter.sum(1))[0]
        p_vs_f = p_vs_f[p_selected]
        p_vs_a = p_vs_a[p_selected]
        p_vs_t = p_vs_t[p_selected]
        p_vs_c = p_vs_c[p_selected]

        # Construct edge indices
        edge_index_pa = np.vstack(p_vs_a.nonzero())
        edge_index_ap = edge_index_pa[[1, 0]]
        edge_index_pf = np.vstack(p_vs_f.nonzero())
        edge_index_fp = edge_index_pf[[1, 0]]

        # Create node features dictionary
        features = tlx.convert_to_tensor(p_vs_t.toarray(), dtype=tlx.float32)
        features_dict = {'paper': features}

        # Process labels
        labels = np.zeros(p_vs_c.shape[0], dtype=np.int64)  # Ensure labels array size matches the number of rows in p_vs_c
        for conf_id, label_id in zip(conf_ids, label_ids):
            if sp.issparse(p_vs_c):
                indices = p_vs_c[:, conf_id].nonzero()[0]
            else:
                indices = np.where(p_vs_c[:, conf_id])[0]
            # Ensure indices are within the bounds of the labels array
            valid_indices = indices[indices < p_vs_c.shape[0]]
            if len(valid_indices) != len(indices):
                print(f"Warning: Some indices are out of bounds and have been ignored. Out-of-bounds indices: {indices[indices >= p_vs_c.shape[0]]}")
            labels[valid_indices] = label_id

        labels = tlx.convert_to_tensor(labels, dtype=tlx.int64)

        num_classes = 3

        # Create train, validation, and test indices
        float_mask = np.zeros(p_vs_c.shape[0], dtype=np.float32)  # float_mask size should match the number of rows in p_vs_c
        for conf_id in conf_ids:
            if sp.issparse(p_vs_c):
                pc_c_mask = (p_vs_c[:, conf_id].toarray().flatten() > 0)
            else:
                pc_c_mask = (p_vs_c[:, conf_id] > 0)
            # Assign random values to papers for each conf_id
            float_mask[pc_c_mask] = np.random.uniform(0, 1, size=pc_c_mask.sum())

        train_idx = np.where(float_mask <= 0.2)[0]
        val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
        test_idx = np.where(float_mask > 0.3)[0]

        num_nodes = features.shape[0]
        train_mask = np.zeros(num_nodes, dtype=bool)
        train_mask[train_idx] = True
        val_mask = np.zeros(num_nodes, dtype=bool)
        val_mask[val_idx] = True
        test_mask = np.zeros(num_nodes, dtype=bool)
        test_mask[test_idx] = True

        # Create the heterogeneous graph
        graph = HeteroGraph()
        graph['paper'].x = features_dict['paper']
        graph['paper'].num_nodes = num_nodes
        graph['author'].num_nodes = p_vs_a.shape[1]
        graph['field'].num_nodes = p_vs_f.shape[1]

        # Add edges
        graph['paper', 'pa', 'author'].edge_index = edge_index_pa
        graph['author', 'ap', 'paper'].edge_index = edge_index_ap
        graph['paper', 'pf', 'field'].edge_index = edge_index_pf
        graph['field', 'fp', 'paper'].edge_index = edge_index_fp

        # Assign labels and masks to paper nodes
        graph['paper'].y = labels
        graph['paper'].train_mask = train_mask
        graph['paper'].val_mask = val_mask
        graph['paper'].test_mask = test_mask

        # Apply pre-transform if defined
        if self.pre_transform is not None:
            graph = self.pre_transform(graph)

        # Save the processed data
        self.save_data(self.collate([graph]), self.processed_paths[0])
