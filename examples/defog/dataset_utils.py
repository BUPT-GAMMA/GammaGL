import numpy as np
import tensorlayerx as tlx
from gammagl.data import Graph

def create_synthetic_dataset(num_graphs=100, min_nodes=10, max_nodes=20,
                              num_node_types=2, num_edge_types=2, p_edge=0.3):
    r"""Create a synthetic graph dataset for testing.

    Parameters
    ----------
    num_graphs : int
        Number of graphs.
    min_nodes, max_nodes : int
        Range of nodes per graph.
    num_node_types : int
        Number of node categories.
    num_edge_types : int
        Number of edge categories (including 'no-edge' at index 0).
    p_edge : float
        Probability of edge existence.

    Returns
    -------
    list
        List of ``Graph`` objects.
    """
    graphs = []
    for _ in range(num_graphs):
        n = np.random.randint(min_nodes, max_nodes + 1)

        # Node features (one-hot)
        node_labels = np.random.randint(0, num_node_types, size=n)
        x = np.eye(num_node_types, dtype=np.float32)[node_labels]

        # Edges: random with probability p_edge
        adj = (np.random.rand(n, n) < p_edge).astype(np.float32)
        adj = np.triu(adj, k=1)
        adj = adj + adj.T
        np.fill_diagonal(adj, 0)

        # Convert to sparse edge_index and edge_attr
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



def compute_dataset_infos(graphs, num_node_types, num_edge_types):
    r"""Compute dataset statistics.

    Parameters
    ----------
    graphs : list
        List of Graph objects.
    num_node_types : int
        Number of node types.
    num_edge_types : int
        Number of edge types.

    Returns
    -------
    dict
        Dataset information dict.
    """
    total_graphs = len(graphs)
    node_counts = []
    node_type_counts = np.zeros(num_node_types, dtype=np.float32)
    edge_type_counts = np.zeros(num_edge_types, dtype=np.float32)

    for idx, g in enumerate(graphs):
        x_val = g.x
        x_np = x_val if isinstance(x_val, np.ndarray) else tlx.convert_to_numpy(x_val)
        n = x_np.shape[0]
        node_counts.append(n)

        node_labels = np.argmax(x_np, axis=-1)
        for label in node_labels:
            node_type_counts[label] += 1

        if g.edge_attr is not None:
            ea_val = g.edge_attr
            ea_np = ea_val if isinstance(ea_val, np.ndarray) else tlx.convert_to_numpy(ea_val)
            edge_type_sums = ea_np.sum(axis=0)
            if edge_type_sums.shape[0] > 1:
                edge_type_counts[1:] += edge_type_sums[1:]

        # Count no-edge pairs into channel 0 only, matching original DeFoG
        total_pairs = n * (n - 1)
        n_edges = g.edge_index.shape[1] if g.edge_index is not None else 0
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



def load_real_dataset(name, root=None, conditional=False, target='mu', remove_h=None):
    r"""Load a real graph generation dataset.

    Parameters
    ----------
    name : str
        Dataset name: ``'planar'``, ``'tree'``, ``'sbm'``, ``'qm9'``,
        ``'guacamol'``, ``'zinc250k'``, or ``'moses'``.
    root : str, optional
        Root directory for dataset storage.
    conditional : bool
        Whether to load conditional labels (only for qm9).
    target : str
        Target property for conditional generation: ``'mu'``, ``'homo'``, or ``'both'``.

    Returns
    -------
    tuple
        ``(train_graphs, val_graphs, test_graphs, dataset_infos, num_node_types, num_edge_types, test_labels)``
    """
    if name == 'planar':
        from datasets.spectre_dataset import PlanarGraphDataset
        ds_cls = PlanarGraphDataset
        num_node_types, num_edge_types = 2, 2
        # Planar/Tree/SBM: x is ones(n,1), need to convert to one-hot for DeFoG
        convert_spectre = True
    elif name == 'tree':
        from datasets.spectre_dataset import TreeGraphDataset
        ds_cls = TreeGraphDataset
        num_node_types, num_edge_types = 2, 2
        convert_spectre = True
    elif name == 'sbm':
        from datasets.spectre_dataset import SBMGraphDataset
        ds_cls = SBMGraphDataset
        num_node_types, num_edge_types = 2, 2
        convert_spectre = True
    elif name == 'comm20':
        from datasets.spectre_dataset import Comm20GraphDataset
        ds_cls = Comm20GraphDataset
        num_node_types, num_edge_types = 2, 2
        convert_spectre = True
    elif name == 'qm9':
        from datasets.qm9_dataset import QM9Gen
        ds_cls = QM9Gen
        qm9_remove_h = True if remove_h is None else bool(remove_h)
        num_node_types, num_edge_types = (4, 5) if qm9_remove_h else (5, 5)
        convert_spectre = False
    elif name == 'guacamol':
        from datasets.guacamol_dataset import GuacaMolDataset
        ds_cls = GuacaMolDataset
        num_node_types, num_edge_types = 12, 5
        convert_spectre = False
    elif name == 'zinc250k':
        from datasets.zinc250k_dataset import ZINC250kGen
        ds_cls = ZINC250kGen
        num_node_types, num_edge_types = 9, 4
        convert_spectre = False
    elif name == 'moses':
        from datasets.moses_dataset import MOSESDataset
        ds_cls = MOSESDataset
        num_node_types, num_edge_types = 8, 5
        convert_spectre = False
    elif name == 'tls':
        from datasets.tls_dataset import TLSGraphDataset
        ds_cls = TLSGraphDataset
        num_node_types, num_edge_types = 9, 2
        convert_spectre = False
    else:
        raise ValueError(f"Unknown dataset: {name}. "
                         f"Use 'synthetic', 'planar', 'tree', 'sbm', 'comm20', 'qm9', "
                         f"'guacamol', 'zinc250k', 'moses', or 'tls'.")

    kwargs = {'root': root} if root else {}
    if name == 'qm9':
        if remove_h is None:
            remove_h = True
        kwargs['remove_h'] = remove_h
        kwargs['aromatic'] = True
    # QM9 and TLS support conditional generation
    if name in ('qm9', 'tls') and conditional:
        kwargs['conditional'] = True
        kwargs['target'] = target
    train_ds = ds_cls(split='train', **kwargs)
    val_ds = ds_cls(split='val', **kwargs)
    test_ds = ds_cls(split='test', **kwargs)

    print(f"Dataset {name}: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Convert spectre datasets: x=ones(n,1) -> one-hot node type
    # edge_attr=[0,1] is already 2-class one-hot (no-edge, edge)
    if convert_spectre:
        train_graphs = []
        for i in range(len(train_ds)):
            g = train_ds[i]
            n = g.x.shape[0]
            # Node features: all same type -> one-hot [1, 0] for type 0
            x_np = np.zeros((n, num_node_types), dtype=np.float32)
            x_np[:, 0] = 1.0
            g_new = Graph(
                x=tlx.convert_to_tensor(x_np, dtype=tlx.float32),
                edge_index=g.edge_index,
                edge_attr=g.edge_attr,
                y=tlx.convert_to_tensor(np.zeros(1, dtype=np.float32), dtype=tlx.float32),
            )
            train_graphs.append(g_new)
    else:
        total_train = len(train_ds)
        train_graphs = []
        for i in range(total_train):
            g = train_ds[i]
            # Preserve y from dataset (conditional labels or empty)
            y_val = g.y if g.y is not None else np.zeros((1, 0), dtype=np.float32)
            if isinstance(y_val, np.ndarray):
                y_val = tlx.convert_to_tensor(y_val, dtype=tlx.float32)
            g_new = Graph(
                x=g.x,
                edge_index=g.edge_index,
                edge_attr=g.edge_attr,
                y=y_val,
            )
            train_graphs.append(g_new)
            if (i + 1) <= 3 or (i + 1) % 50000 == 0 or (i + 1) == total_train:
                print(f"  Loading graph {i + 1}/{total_train}...")
            # Periodic GPU cache flush to avoid OOM on large datasets
            if tlx.BACKEND == 'torch' and (i + 1) % 10000 == 0:
                import torch as _torch
                _torch.cuda.empty_cache()

    if name in ('qm9', 'guacamol', 'zinc250k', 'moses'):
        dataset_infos = {
            'output_dims': {
                'X': num_node_types,
                'E': num_edge_types,
                'y': 0,
            }
        }
        if name == 'qm9':
            qm9_remove_h = bool(getattr(train_ds, 'remove_h', kwargs.get('remove_h', True)))
            stats = QM9Gen.STATS_REMOVE_H if qm9_remove_h else QM9Gen.STATS_WITH_H
            atom_decoder = [stats['atom_names'][i] for i in range(len(stats['atom_names']))]
            dataset_infos['node_types'] = stats['node_types'].astype(np.float32).copy()
            dataset_infos['edge_types'] = stats['edge_types'].astype(np.float32).copy()
            dataset_infos['node_dist'] = stats['n_nodes'].astype(np.float32).copy()
            dataset_infos['max_n_nodes'] = int(stats['max_n_nodes'])
            dataset_infos['remove_h'] = qm9_remove_h
            dataset_infos['valencies'] = list(stats['valencies'])
            dataset_infos['valency_distribution'] = (
                stats['valency_distribution'].astype(np.float32).copy()
            )
            dataset_infos['atom_weights'] = dict(stats['atom_weights'])
            dataset_infos['max_weight'] = float(stats['max_weight'])
        elif name == 'guacamol':
            atom_decoder = ds_cls.ATOM_DECODER
            stats = ds_cls.STATS
            dataset_infos['node_types'] = stats['node_types'].astype(np.float32).copy()
            dataset_infos['edge_types'] = stats['edge_types'].astype(np.float32).copy()
            dataset_infos['remove_h'] = True
            dataset_infos['valencies'] = list(stats['valencies'])
            dataset_infos['atom_weights'] = dict(stats['atom_weights'])
            dataset_infos['max_weight'] = float(stats['max_weight'])
        elif name == 'zinc250k':
            atom_decoder = ds_cls.ATOM_DECODER
            stats = ds_cls.STATS
            dataset_infos['node_types'] = stats['node_types'].astype(np.float32).copy()
            dataset_infos['edge_types'] = stats['edge_types'].astype(np.float32).copy()
            dataset_infos['node_dist'] = stats['n_nodes'].astype(np.float32).copy()
            dataset_infos['max_n_nodes'] = int(stats['max_n_nodes'])
            dataset_infos['remove_h'] = True
            dataset_infos['valencies'] = list(stats['valencies'])
            dataset_infos['valency_distribution'] = (
                stats['valency_distribution'].astype(np.float32).copy()
            )
            dataset_infos['atom_weights'] = dict(stats['atom_weights'])
            dataset_infos['max_weight'] = float(stats['max_weight'])
        elif name == 'moses':
            atom_decoder = ds_cls.ATOM_DECODER
            stats = ds_cls.STATS
            dataset_infos['node_types'] = stats['node_types'].astype(np.float32).copy()
            dataset_infos['edge_types'] = stats['edge_types'].astype(np.float32).copy()
            dataset_infos['node_dist'] = stats['n_nodes'].astype(np.float32).copy()
            dataset_infos['max_n_nodes'] = int(stats['n_nodes'].shape[0] - 1)
            dataset_infos['remove_h'] = False
            dataset_infos['valencies'] = list(stats['valencies'])
            dataset_infos['atom_weights'] = dict(stats['atom_weights'])
            dataset_infos['max_weight'] = float(stats['max_weight'])
        dataset_infos['atom_decoder'] = list(atom_decoder)
    else:
        dataset_infos = compute_dataset_infos(train_graphs, num_node_types, num_edge_types)

    # Collect test labels for conditional sampling
    test_labels = None
    if conditional:
        test_labels_list = []
        for i in range(len(test_ds)):
            g = test_ds[i]
            y_val = g.y if g.y is not None else np.zeros((1, 0), dtype=np.float32)
            if isinstance(y_val, np.ndarray):
                y_np = y_val.flatten()
            else:
                y_np = tlx.convert_to_numpy(y_val).flatten()
            test_labels_list.append(y_np)
        if len(test_labels_list) > 0 and len(test_labels_list[0]) > 0:
            test_labels = tlx.convert_to_tensor(
                np.stack(test_labels_list, axis=0).astype(np.float32))

    return train_graphs, val_ds, test_ds, dataset_infos, num_node_types, num_edge_types, test_labels
