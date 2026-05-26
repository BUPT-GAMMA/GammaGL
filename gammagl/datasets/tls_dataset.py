"""
TLS (Tertiary Lymphoid Structures) graph dataset for conditional generation.

Ported from DeFoG src/datasets/tls_dataset.py for GammaGL.

Contains cell graphs from histopathology images, where each node represents
a cell with a phenotype label. Graph-level labels indicate TLS content
(high vs low) for conditional generation.

Reference: https://arxiv.org/pdf/2310.06661.pdf
"""
import os
import os.path as osp
import copy
import pickle
import numpy as np
import networkx as nx
import tensorlayerx as tlx
from typing import Callable, List, Optional

from gammagl.data import (
    Graph,
    InMemoryDataset,
    download_url,
)


# ============================================================
# Cell phenotype encoding
# ============================================================

PHENOTYPE_DECODER = [
    "B",
    "T",
    "Epithelial",
    "Fibroblast",
    "Myofibroblast",
    "CD38+ Lymphocyte",
    "Macrophages/Granulocytes",
    "Marker",
    "Endothelial",
]

PHENOTYPE_ENCODER = {v: k for k, v in enumerate(PHENOTYPE_DECODER)}

# For conditional generation: 9 node types, 2 edge types
TLS_NUM_NODE_TYPES = 9
TLS_NUM_EDGE_TYPES = 2


# ============================================================
# CellGraph: networkx-based cell graph with TLS features
# ============================================================

class CellGraph(nx.Graph):
    """A networkx Graph augmented with TLS feature computation.

    Each node must have a 'phenotype' attribute.
    """

    def __init__(self, graph):
        super().__init__()
        self.add_nodes_from(graph.nodes(data=True))
        self.add_edges_from(graph.edges(data=True))
        self.tls_features = self.compute_tls_features()

    def has_low_TLS(self):
        return self.tls_features["k_1"] < 0.05

    def has_high_TLS(self):
        return 0.05 < self.tls_features["k_2"]

    def to_label(self):
        """Return binary label: 0=low TLS, 1=high TLS, -1=ambiguous."""
        if self.has_low_TLS():
            return 0
        elif self.has_high_TLS():
            return 1
        else:
            return -1

    def to_gamma_graph(self):
        """Convert to GammaGL Graph object."""
        n = self.number_of_nodes()
        phenotypes = [self.nodes[node].get("phenotype", "Marker") for node in self.nodes()]
        encoded = [PHENOTYPE_ENCODER.get(p, 7) for p in phenotypes]  # default to Marker

        x = np.eye(TLS_NUM_NODE_TYPES, dtype=np.float32)[encoded]

        adj = nx.to_numpy_array(self).astype(np.float32)
        np.fill_diagonal(adj, 0)
        src, dst = np.nonzero(adj)
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)
        # All edges are type 1 (edge present)
        edge_attr = np.zeros((len(src), TLS_NUM_EDGE_TYPES), dtype=np.float32)
        edge_attr[:, 1] = 1.0

        data = Graph(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=np.zeros((1, 0), dtype=np.float32),  # placeholder, filled by transform
            n_nodes=np.array([n], dtype=np.int64),
            to_tensor=True,
        )
        return data

    @classmethod
    def from_dense_graph(cls, dense_graph):
        """Construct from (node_features, adjacency_matrix) tuple.

        Parameters
        ----------
        dense_graph : tuple of (node_features, adjacency_matrix)
            node_features: array of int (n,), adjacency_matrix: array of int (n,n)

        Returns
        -------
        CellGraph
        """
        node_features = dense_graph[0]
        adj = dense_graph[1]
        if not isinstance(node_features, np.ndarray):
            node_features = np.asarray(node_features).astype(int)
        if not isinstance(adj, np.ndarray):
            adj = np.asarray(adj).astype(int)

        # Filter out padding nodes (type == -1)
        n_nodes = int(np.sum(node_features != -1))
        node_features = node_features[:n_nodes]
        adj = adj[:n_nodes, :n_nodes]

        nx_graph = nx.from_numpy_array(adj)
        for _, _, data in nx_graph.edges(data=True):
            del data["weight"]

        for node_idx in nx_graph.nodes():
            encoded_phenotype = int(node_features[node_idx])
            phenotype = PHENOTYPE_DECODER[encoded_phenotype] if encoded_phenotype < len(PHENOTYPE_DECODER) else "Marker"
            nx_graph.nodes[node_idx]["phenotype"] = phenotype

        return cls(nx_graph)

    # ============================================================
    # TLS feature computation
    # ============================================================

    @property
    def map_phenotype_to_color(self):
        return {
            "B": "c",
            "T": "b",
            "Epithelial": "k",
            "Fibroblast": "#C4A484",
            "Myofibroblast": "g",
            "CD38+ Lymphocyte": "#FEE12B",
            "Macrophages/Granulocytes": "C3",
            "Marker": "0.9",
            "Endothelial": "0.75",
        }

    def classify_TLS_edge(self, edge):
        allowed_cell_types = ["B", "T"]
        start_node, end_node = edge
        start_phenotype = self.nodes[start_node]["phenotype"]
        end_phenotype = self.nodes[end_node]["phenotype"]

        if start_phenotype not in allowed_cell_types or end_phenotype not in allowed_cell_types:
            edge_type = "ignore"
        elif start_phenotype == end_phenotype:
            edge_type = "alpha"
        else:
            b_cell = start_node if start_phenotype == "B" else end_node
            num_of_b_neighbors = len([
                node for node in self.neighbors(b_cell)
                if self.nodes[node]["phenotype"] == "B"
            ])
            edge_type = f"gamma_{num_of_b_neighbors}"
        return edge_type

    def compute_tls_features(self, a_max=5, min_num_gamma_edges=0):
        """Compute TLS feature metric from arxiv.org/pdf/2310.06661.pdf."""
        if self.is_directed():
            raise ValueError("Graph should be undirected.")

        graph_phenotypes = nx.get_node_attributes(self, "phenotype")
        nodes_to_remove = [
            node for node, phenotype in graph_phenotypes.items()
            if phenotype != "B" and phenotype != "T"
        ]
        bt_subgraph = copy.deepcopy(self)
        bt_subgraph.remove_nodes_from(nodes_to_remove)
        total_num_edges = bt_subgraph.number_of_edges()

        num_edge_types_idxs = self._get_edges_idxs_by_tlo_type(a_max)

        denominator = total_num_edges - num_edge_types_idxs["alpha"]
        if denominator < min_num_gamma_edges:
            return {f"k_{a}": None for a in range(a_max + 1)}

        tlo_dict = {}
        if denominator == 0:
            tlo_dict.update({f"k_{a}": 0.0 for a in range(a_max + 1)})
        else:
            k = 1.0
            for a in range(a_max + 1):
                k -= num_edge_types_idxs[f"gamma_{a}"] / denominator
                tlo_dict.update({f"k_{a}": max(0.0, round(k, 4))})
        return tlo_dict

    def _get_edges_idxs_by_tlo_type(self, a_max):
        num_edges_by_type = {f"gamma_{a}": 0 for a in range(a_max + 1)}
        num_edges_by_type["alpha"] = 0
        for edge in self.edges:
            edge_type = self.classify_TLS_edge(edge)
            if edge_type in num_edges_by_type:
                num_edges_by_type[edge_type] += 1
        return num_edges_by_type


# ============================================================
# Dataset transforms
# ============================================================

class SelectK2Transform:
    """Transform: set y to binary label based on k_2 > 0.05."""

    def __call__(self, data):
        # y is set during processing; this is a placeholder for external use
        return data


# ============================================================
# Dataset
# ============================================================

class TLSGraphDataset(InMemoryDataset):
    r"""TLS (Tertiary Lymphoid Structures) graph dataset.

    Contains cell graphs from histopathology images. Each node has a phenotype
    label (B, T, Epithelial, etc.) and the graph-level label indicates TLS content.

    Data is downloaded from the ConStruct repository.

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset should be saved.
    split : str
        Which split to load: ``'train'``, ``'val'``, or ``'test'``.
    conditional : bool
        Whether to include conditional labels (binary TLS class).
    target : str
        Target property: ``'k2'`` (default).
    transform : callable, optional
        A transform applied to each :class:`Graph` on access.
    pre_transform : callable, optional
        A transform applied during processing before saving.
    pre_filter : callable, optional
        A filter applied during processing.
    force_reload : bool
        Whether to re-process the dataset.
    """

    urls = {
        'low_tls': {
            'train': 'https://github.com/manuelmlmadeira/ConStruct/raw/main/data/low_tls_200/raw/train.pkl',
            'val': 'https://github.com/manuelmlmadeira/ConStruct/raw/main/data/low_tls_200/raw/val.pkl',
            'test': 'https://github.com/manuelmlmadeira/ConStruct/raw/main/data/low_tls_200/raw/test.pkl',
        },
        'high_tls': {
            'train': 'https://github.com/manuelmlmadeira/ConStruct/raw/main/data/high_tls_200/raw/train.pkl',
            'val': 'https://github.com/manuelmlmadeira/ConStruct/raw/main/data/high_tls_200/raw/val.pkl',
            'test': 'https://github.com/manuelmlmadeira/ConStruct/raw/main/data/high_tls_200/raw/test.pkl',
        },
    }

    def __init__(self, root: Optional[str] = None, split: str = 'train',
                 conditional: bool = False, target: str = 'k2',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 force_reload: bool = False):
        assert split in ('train', 'val', 'test'), f"Unknown split: {split}"
        self.name = 'tls'
        self.split = split
        self.conditional = conditional
        self.target = target

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        split_idx = {'train': 0, 'val': 1, 'test': 2}[split]
        self.data, self.slices = self.load_data(self.processed_paths[split_idx])

    @property
    def raw_file_names(self) -> List[str]:
        return ['train.pkl', 'val.pkl', 'test.pkl']

    @property
    def processed_file_names(self) -> List[str]:
        tag = '_cond' if self.conditional else ''
        return [
            tlx.BACKEND + f'_train{tag}.pt',
            tlx.BACKEND + f'_val{tag}.pt',
            tlx.BACKEND + f'_test{tag}.pt',
        ]

    def download(self):
        for split_key in ['train', 'val', 'test']:
            low_path = download_url(
                self.urls['low_tls'][split_key],
                osp.join(self.raw_dir, 'low_tls'))
            low_data = pickle.load(open(low_path, 'rb'))

            high_path = download_url(
                self.urls['high_tls'][split_key],
                osp.join(self.raw_dir, 'high_tls'))
            high_data = pickle.load(open(high_path, 'rb'))

            all_data = low_data + high_data
            with open(osp.join(self.raw_dir, f'{split_key}.pkl'), 'wb') as f:
                pickle.dump(all_data, f)
            print(f"  TLS {split_key}: {len(all_data)} graphs")

    def process(self):
        for split_name, split_idx in [('train', 0), ('val', 1), ('test', 2)]:
            pkl_path = osp.join(self.raw_dir, f'{split_name}.pkl')
            with open(pkl_path, 'rb') as f:
                raw_dataset = pickle.load(f)

            data_list = []
            for idx, graph in enumerate(raw_dataset):
                cell_graph = CellGraph(graph)

                if not (cell_graph.has_low_TLS() or cell_graph.has_high_TLS()):
                    print(f"  Warning: ambiguous cell graph {idx}, skipping")
                    continue

                data = cell_graph.to_gamma_graph()

                # Set conditional label
                if self.conditional and self.target == 'k2':
                    label = float(cell_graph.tls_features.get("k_2", 0) > 0.05)
                    data.y = np.array([[label]], dtype=np.float32)

                # Store TLS features as additional attribute
                data.tls_features = np.array(
                    [cell_graph.tls_features.get(f"k_{a}", 0) or 0 for a in range(6)],
                    dtype=np.float32
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            print(f"  Processed TLS {split_name}: {len(data_list)} graphs")
            collated_data, slices = self.collate(data_list)
            self.save_data(
                (collated_data, slices),
                self.processed_paths[split_idx],
            )

    def __repr__(self) -> str:
        return f'TLSGraphDataset(split={self.split}, len={len(self)})'
