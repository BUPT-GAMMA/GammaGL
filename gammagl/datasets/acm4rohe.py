import os.path as osp
import numpy as np
import tensorlayerx as tlx
from gammagl.data import InMemoryDataset, HeteroGraph, download_url
from scipy import io as sio
from typing import Callable, List, Optional

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
        p_vs_c = data["PvsC"]  # Paper-conference labels

        # Create heterogeneous graph
        graph = HeteroGraph()
        graph['paper'].x = tlx.convert_to_tensor(p_vs_t.toarray(), dtype=tlx.float32)  # Paper features
        graph['paper'].num_nodes = p_vs_t.shape[0]
        graph['author'].num_nodes = p_vs_a.shape[1]
        graph['field'].num_nodes = p_vs_f.shape[1]

        # Define edges for paper-author and paper-field relationships
        edge_index_pa = np.vstack(p_vs_a.nonzero())
        edge_index_ap = edge_index_pa[[1, 0]]
        edge_index_pf = np.vstack(p_vs_f.nonzero())
        edge_index_fp = edge_index_pf[[1, 0]]
        
        # Assign edge indices
        graph['paper', 'pa', 'author'].edge_index = tlx.convert_to_tensor(edge_index_pa)
        graph['author', 'ap', 'paper'].edge_index = tlx.convert_to_tensor(edge_index_ap)
        graph['paper', 'pf', 'field'].edge_index = tlx.convert_to_tensor(edge_index_pf)
        graph['field', 'fp', 'paper'].edge_index = tlx.convert_to_tensor(edge_index_fp)

        # Process labels and masks
        conf_ids = [0, 1, 9, 10, 13]  # Specific conference IDs
        label_ids = [0, 1, 2, 2, 1]
        p_vs_c_filtered = p_vs_c[:, conf_ids]
        p_selected = np.nonzero(p_vs_c_filtered.sum(1))[0]
        
        labels = np.zeros(len(p_selected), dtype=np.int64)
        for conf_id, label_id in zip(conf_ids, label_ids):
            labels[np.where(p_vs_c[:, conf_id])[0]] = label_id
        
        graph['paper'].y = tlx.convert_to_tensor(labels, dtype=tlx.int64)
        
        # Generate train, validation, and test masks
        num_papers = p_vs_t.shape[0]
        train_mask = np.zeros(num_papers, dtype=bool)
        val_mask = np.zeros(num_papers, dtype=bool)
        test_mask = np.zeros(num_papers, dtype=bool)

        float_mask = np.random.rand(num_papers)
        train_mask[float_mask < 0.2] = True
        val_mask[(float_mask >= 0.2) & (float_mask < 0.3)] = True
        test_mask[float_mask >= 0.3] = True

        graph['paper'].train_mask = train_mask
        graph['paper'].val_mask = val_mask
        graph['paper'].test_mask = test_mask

        # Apply pre-transform if defined
        if self.pre_transform is not None:
            graph = self.pre_transform(graph)
        
        # Save the processed data
        self.save_data(self.collate([graph]), self.processed_paths[0])
