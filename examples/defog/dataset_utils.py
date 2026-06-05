import numpy as np
import tensorlayerx as tlx
from gammagl.data import Graph
from gammagl.datasets import (
    PlanarGraphDataset, TreeGraphDataset, SBMGraphDataset, Comm20GraphDataset,
    QM9Gen, GuacaMolDataset, ZINC250kGen, MOSESDataset, TLSGraphDataset
)

def create_synthetic_dataset(num_graphs=100, min_nodes=10, max_nodes=20,
                              num_node_types=2, num_edge_types=2, p_edge=0.3):
    r"""Create a synthetic graph dataset for testing."""
    graphs = []
    for _ in range(num_graphs):
        n = np.random.randint(min_nodes, max_nodes + 1)
        node_labels = np.random.randint(0, num_node_types, size=n)
        x = np.eye(num_node_types, dtype=np.float32)[node_labels]
        adj = (np.random.rand(n, n) < p_edge).astype(np.float32)
        adj = np.triu(adj, k=1)
        adj = adj + adj.T
        np.fill_diagonal(adj, 0)
        src, dst = np.nonzero(adj)
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)

        if num_edge_types > 1:
            edge_labels = np.random.randint(1, num_edge_types, size=len(src))
            edge_attr = np.eye(num_edge_types, dtype=np.float32)[edge_labels]
        else:
            edge_attr = np.ones((len(src), 1), dtype=np.float32)

        g = Graph(
            x=tlx.convert_to_tensor(x),
            edge_index=tlx.convert_to_tensor(edge_index),
            edge_attr=tlx.convert_to_tensor(edge_attr),
            y=tlx.convert_to_tensor(np.zeros(1, dtype=np.float32)),
        )
        graphs.append(g)
    return graphs


def compute_dataset_infos(dataset, num_node_types, num_edge_types):
    r"""Compute dataset statistics dynamically."""
    total_graphs = len(dataset)
    node_counts = []
    node_type_counts = np.zeros(num_node_types, dtype=np.float32)
    edge_type_counts = np.zeros(num_edge_types, dtype=np.float32)

    for idx in range(total_graphs):
        g = dataset[idx]
        x_val = g.x
        x_np = x_val if isinstance(x_val, np.ndarray) else tlx.convert_to_numpy(x_val)
        n = x_np.shape[0]
        node_counts.append(n)

        node_labels = np.argmax(x_np, axis=-1)
        for label in node_labels:
            node_type_counts[label] += 1

        if getattr(g, 'edge_attr', None) is not None:
            ea_val = g.edge_attr
            ea_np = ea_val if isinstance(ea_val, np.ndarray) else tlx.convert_to_numpy(ea_val)
            edge_type_sums = ea_np.sum(axis=0)
            if edge_type_sums.shape[0] > 1:
                edge_type_counts[1:] += edge_type_sums[1:]

        total_pairs = n * (n - 1)
        n_edges = g.edge_index.shape[1] if getattr(g, 'edge_index', None) is not None else 0
        n_no_edge = total_pairs - n_edges
        edge_type_counts[0] += n_no_edge

        if (idx + 1) <= 3 or (idx + 1) % 50000 == 0 or (idx + 1) == total_graphs:
            print(f"  Processing graph {idx + 1}/{total_graphs}...")

    max_n = max(node_counts)
    node_dist = np.zeros(max_n + 1, dtype=np.float32)
    for nc in node_counts:
        node_dist[nc] += 1
    node_dist = node_dist / node_dist.sum()

    return {
        'output_dims': {
            'X': num_node_types,
            'E': num_edge_types,
            'y': 0,
        },
        'node_types': node_type_counts,
        'edge_types': edge_type_counts,
        'max_n_nodes': max_n,
        'node_dist': node_dist,
    }


class SpectreNodeTransform:
    def __init__(self, num_node_types=2):
        self.num_node_types = num_node_types

    def __call__(self, data):
        n = data.x.shape[0]
        x_np = np.zeros((n, self.num_node_types), dtype=np.float32)
        x_np[:, 0] = 1.0
        data.x = tlx.convert_to_tensor(x_np, dtype=tlx.float32)
        y_val = data.y if hasattr(data, 'y') and data.y is not None else np.zeros((1, 0), dtype=np.float32)
        if isinstance(y_val, np.ndarray) and y_val.size == 0:
            y_val = np.zeros((1, 0), dtype=np.float32)
        data.y = tlx.convert_to_tensor(y_val, dtype=tlx.float32) if isinstance(y_val, np.ndarray) else y_val
        return data


class GenericNodeTransform:
    def __call__(self, data):
        y_val = data.y if hasattr(data, 'y') and data.y is not None else np.zeros((1, 0), dtype=np.float32)
        data.y = tlx.convert_to_tensor(y_val, dtype=tlx.float32) if isinstance(y_val, np.ndarray) else y_val
        return data


def load_real_dataset(name, root=None, conditional=False, target='mu', remove_h=None):
    if name == 'planar':
        ds_cls = PlanarGraphDataset
        num_node_types, num_edge_types = 2, 2
        convert_spectre = True
    elif name == 'tree':
        ds_cls = TreeGraphDataset
        num_node_types, num_edge_types = 2, 2
        convert_spectre = True
    elif name == 'sbm':
        ds_cls = SBMGraphDataset
        num_node_types, num_edge_types = 2, 2
        convert_spectre = True
    elif name == 'comm20':
        ds_cls = Comm20GraphDataset
        num_node_types, num_edge_types = 2, 2
        convert_spectre = True
    elif name == 'qm9':
        ds_cls = QM9Gen
        qm9_remove_h = True if remove_h is None else bool(remove_h)
        num_node_types, num_edge_types = (4, 5) if qm9_remove_h else (5, 5)
        convert_spectre = False
    elif name == 'guacamol':
        ds_cls = GuacaMolDataset
        num_node_types, num_edge_types = 12, 5
        convert_spectre = False
    elif name == 'zinc250k':
        ds_cls = ZINC250kGen
        num_node_types, num_edge_types = 9, 4
        convert_spectre = False
    elif name == 'moses':
        ds_cls = MOSESDataset
        num_node_types, num_edge_types = 8, 5
        convert_spectre = False
    elif name == 'tls':
        ds_cls = TLSGraphDataset
        num_node_types, num_edge_types = 9, 2
        convert_spectre = False
    else:
        raise ValueError(f"Unknown dataset: {name}")

    kwargs = {'root': root} if root else {}
    if name == 'qm9':
        kwargs['remove_h'] = True if remove_h is None else remove_h
        kwargs['aromatic'] = True
    if name in ('qm9', 'tls') and conditional:
        kwargs['conditional'] = True
        kwargs['target'] = target

    # Inject transform
    transform = SpectreNodeTransform(num_node_types) if convert_spectre else GenericNodeTransform()
    kwargs['transform'] = transform

    train_ds = ds_cls(split='train', **kwargs)
    val_ds = ds_cls(split='val', **kwargs)
    test_ds = ds_cls(split='test', **kwargs)

    print(f"Dataset {name}: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    if name in ('qm9', 'guacamol', 'zinc250k', 'moses'):
        dataset_infos = {'output_dims': {'X': num_node_types, 'E': num_edge_types, 'y': 0}}
        stats = ds_cls.STATS_REMOVE_H if name == 'qm9' and kwargs.get('remove_h') else getattr(ds_cls, 'STATS_WITH_H', getattr(ds_cls, 'STATS', None))
        
        if stats:
            atom_decoder = stats.get('atom_names', getattr(ds_cls, 'ATOM_DECODER', []))
            dataset_infos['node_types'] = stats['node_types'].astype(np.float32).copy()
            dataset_infos['edge_types'] = stats['edge_types'].astype(np.float32).copy()
            if 'n_nodes' in stats:
                dataset_infos['node_dist'] = stats['n_nodes'].astype(np.float32).copy()
                dataset_infos['max_n_nodes'] = int(stats.get('max_n_nodes', len(stats['n_nodes']) - 1))
            dataset_infos['remove_h'] = kwargs.get('remove_h', True)
            dataset_infos['valencies'] = list(stats['valencies'])
            if 'valency_distribution' in stats:
                dataset_infos['valency_distribution'] = stats['valency_distribution'].astype(np.float32).copy()
            dataset_infos['atom_weights'] = dict(stats['atom_weights'])
            dataset_infos['max_weight'] = float(stats['max_weight'])
            dataset_infos['atom_decoder'] = list(atom_decoder)
    else:
        dataset_infos = compute_dataset_infos(train_ds, num_node_types, num_edge_types)

    test_labels = None
    if conditional:
        test_labels_list = [tlx.convert_to_numpy(g.y).flatten() for g in test_ds]
        if test_labels_list and len(test_labels_list[0]) > 0:
            test_labels = tlx.convert_to_tensor(np.stack(test_labels_list, axis=0).astype(np.float32))

    return train_ds, val_ds, test_ds, dataset_infos, num_node_types, num_edge_types, test_labels
