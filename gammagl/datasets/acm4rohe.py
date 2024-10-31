import os
import os.path as osp
import numpy as np
import tensorlayerx as tlx
from gammagl.data import InMemoryDataset, HeteroGraph, download_url
from scipy import io as sio
from typing import Callable, List, Optional
import scipy.sparse as sp

class ACM4Rohe(InMemoryDataset):
    r"""The ACM dataset for heterogeneous graph neural networks, consisting of nodes of types
    :obj:`"paper"`, :obj:`"author"`, and :obj:`"field"`. This dataset was adapted from
    `"Heterogeneous Graph Attention Network" <https://github.com/Jhy1993/HAN>`_,
    and is typically used for semi-supervised node classification in
    heterogeneous graphs.

    Parameters
    ----------
    root: str, optional
        Root directory where the dataset should be saved.
    transform: callable, optional
        A function/transform that takes in an :obj:`HeteroGraph` object
        and returns a transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform: callable, optional
        A function/transform that takes in an :obj:`HeteroGraph` object
        and returns a transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload: bool, optional
        Whether to re-process the dataset, even if it has already been processed.
        (default: :obj:`False`)

    Attributes
    ----------
    url: str
        URL where the raw ACM data file can be downloaded.
    """

    url = "https://github.com/Jhy1993/HAN/raw/master/data/acm/ACM.mat"

    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, force_reload: bool = False):
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "ACM.mat",
            "data/generated_attacks/adv_acm_pap_pa_1.pkl",
            "data/generated_attacks/adv_acm_pap_pa_3.pkl",
            "data/generated_attacks/adv_acm_pap_pa_5.pkl",
            "data/preprocess/target_nodes/acm_r_target0.pkl",
            "data/preprocess/target_nodes/acm_r_target1.pkl",
            "data/preprocess/target_nodes/acm_r_target2.pkl",
            "data/preprocess/target_nodes/acm_r_target3.pkl",
            "data/preprocess/target_nodes/acm_r_target4.pkl"
        ]

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + "_data.pt"

    def download(self):
        # Download the main ACM dataset file
        download_url(self.url, self.raw_dir)

        # Download additional adversarial attack data files if missing
        base_url = "https://raw.githubusercontent.com/BUPT-GAMMA/RoHe/main/Code/data"

        # List of required adversarial files to download
        files_to_download = [
            "generated_attacks/adv_acm_pap_pa_1.pkl",
            "generated_attacks/adv_acm_pap_pa_3.pkl",
            "generated_attacks/adv_acm_pap_pa_5.pkl",
            "preprocess/target_nodes/acm_r_target0.pkl",
            "preprocess/target_nodes/acm_r_target1.pkl",
            "preprocess/target_nodes/acm_r_target2.pkl",
            "preprocess/target_nodes/acm_r_target3.pkl",
            "preprocess/target_nodes/acm_r_target4.pkl"
        ]

        # Download each file if not already present in its designated path
        for file_path in files_to_download:
            file_url = f"{base_url}/{file_path}"
            save_folder = os.path.join(self.raw_dir, "data", os.path.dirname(file_path))
            os.makedirs(save_folder, exist_ok=True)  # Ensure save directory exists

            save_path = os.path.join(save_folder, os.path.basename(file_path))
            if not os.path.exists(save_path):
                download_url(file_url, save_folder)

    def process(self):
        data_path = osp.join(self.raw_dir, "ACM.mat")
        data = sio.loadmat(data_path)

        p_vs_f = data["PvsL"]
        p_vs_a = data["PvsA"]
        p_vs_t = data["PvsT"]
        p_vs_c = data["PvsC"]

        conf_ids = [0, 1, 9, 10, 13]
        label_ids = [0, 1, 2, 2, 1]

        p_vs_c_filter = p_vs_c[:, conf_ids]
        p_selected = np.nonzero(p_vs_c_filter.sum(1))[0]
        p_vs_f = p_vs_f[p_selected]
        p_vs_a = p_vs_a[p_selected]
        p_vs_t = p_vs_t[p_selected]
        p_vs_c = p_vs_c[p_selected]

        edge_index_pa = np.vstack(p_vs_a.nonzero())
        edge_index_ap = edge_index_pa[[1, 0]]
        edge_index_pf = np.vstack(p_vs_f.nonzero())
        edge_index_fp = edge_index_pf[[1, 0]]

        features = tlx.convert_to_tensor(p_vs_t.toarray(), dtype=tlx.float32)
        features_dict = {'paper': features}

        labels = np.zeros(p_vs_c.shape[0], dtype=np.int64)
        for conf_id, label_id in zip(conf_ids, label_ids):
            if sp.issparse(p_vs_c):
                indices = p_vs_c[:, conf_id].nonzero()[0]
            else:
                indices = np.where(p_vs_c[:, conf_id])[0]

            valid_indices = indices[indices < p_vs_c.shape[0]]
            labels[valid_indices] = label_id

        labels = tlx.convert_to_tensor(labels, dtype=tlx.int64)

        num_classes = 3

        float_mask = np.zeros(p_vs_c.shape[0], dtype=np.float32)
        for conf_id in conf_ids:
            if sp.issparse(p_vs_c):
                pc_c_mask = (p_vs_c[:, conf_id].toarray().flatten() > 0)
            else:
                pc_c_mask = (p_vs_c[:, conf_id] > 0)

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

        graph = HeteroGraph()
        graph['paper'].x = features_dict['paper']
        graph['paper'].num_nodes = num_nodes
        graph['author'].num_nodes = p_vs_a.shape[1]
        graph['field'].num_nodes = p_vs_f.shape[1]

        graph['paper', 'pa', 'author'].edge_index = edge_index_pa
        graph['author', 'ap', 'paper'].edge_index = edge_index_ap
        graph['paper', 'pf', 'field'].edge_index = edge_index_pf
        graph['field', 'fp', 'paper'].edge_index = edge_index_fp

        graph['paper'].y = labels
        graph['paper'].train_mask = tlx.convert_to_tensor(train_mask, dtype=tlx.bool)
        graph['paper'].val_mask = tlx.convert_to_tensor(val_mask, dtype=tlx.bool)
        graph['paper'].test_mask = tlx.convert_to_tensor(test_mask, dtype=tlx.bool)
        if self.pre_transform is not None:
            graph = self.pre_transform(graph)

        self.save_data(self.collate([graph]), self.processed_paths[0])

    def get_meta_graph(self, dataname, given_adj_dict, features_dict, labels=None, train_mask=None, val_mask=None, test_mask=None):
        r"""Creates a meta-path based `HeteroGraph` for the ACM dataset.

        This function constructs a `HeteroGraph` with meta-path based edges
        between `paper` nodes, representing the meta-paths:
        - Paper -> Author -> Paper (PAP)
        - Paper -> Field -> Paper (PFP)

        """
        meta_graph = HeteroGraph()
        meta_graph['paper'].x = features_dict['paper']
        meta_graph['paper'].num_nodes = features_dict['paper'].shape[0]

        meta_graph['paper', 'author', 'paper'].edge_index = np.array(given_adj_dict['pa'].dot(given_adj_dict['ap']).nonzero())
        meta_graph['paper', 'field', 'paper'].edge_index = np.array(given_adj_dict['pf'].dot(given_adj_dict['fp']).nonzero())

        meta_graph['paper'].y = labels
        meta_graph['paper'].train_mask = train_mask
        meta_graph['paper'].val_mask = val_mask
        meta_graph['paper'].test_mask = test_mask

        return meta_graph