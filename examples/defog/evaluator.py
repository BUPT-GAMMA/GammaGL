"""Evaluation metrics orchestration for DeFoG.

This module contains the evaluation pipeline extracted from defog_trainer.py,
including molecular metrics (validity, uniqueness, novelty, FCD) and
synthetic graph metrics (SPECTRE MMD-based statistics).
"""

import os
import numpy as np
import tensorlayerx as tlx

from rdkit_functions import compute_molecular_metrics


def compute_selection_score(dataset_name, metrics):
    """Compute a single scalar selection score from evaluation metrics.

    Used for best-checkpoint selection during training validation.
    """
    if dataset_name in ('planar', 'tree', 'sbm', 'comm20'):
        for key in (
            'sampling/frac_unic_non_iso_valid',
            'sampling/frac_unique_non_iso',
            'sampling/frac_non_iso',
            'sampling/frac_unique',
            'frac_unic_non_iso_valid',
            'frac_unique_non_iso',
            'frac_non_iso',
            'frac_unique',
            'valid',
            'planar_acc',
            'tree_acc',
        ):
            if key in metrics:
                return float(metrics[key])
        return float('-inf')

    if dataset_name in ('qm9', 'guacamol', 'zinc250k', 'moses'):
        score_parts = []
        for key in ('Validity', 'Relaxed Validity', 'Uniqueness', 'Novelty'):
            value = metrics.get(key)
            if value is not None and value >= 0:
                score_parts.append(float(value))
        return float(np.mean(score_parts)) if score_parts else float('-inf')

    degree = metrics.get('degree')
    return -float(degree) if degree is not None else float('-inf')


def graphs_to_networkx(graph_list):
    """Convert a list of GammaGL Graph objects to networkx graphs.

    Parameters
    ----------
    graph_list : list of Graph
        Each graph has .x (one-hot node features), .edge_index, .edge_attr.

    Returns
    -------
    list of nx.Graph
    """
    import networkx as nx
    nx_graphs = []
    for g in graph_list:
        x_np = g.x if isinstance(g.x, np.ndarray) else tlx.convert_to_numpy(g.x)
        edge_np = g.edge_index if isinstance(g.edge_index, np.ndarray) else tlx.convert_to_numpy(g.edge_index)

        nx_g = nx.Graph()
        n = x_np.shape[0]
        nx_g.add_nodes_from(range(n))

        if edge_np.shape[1] > 0:
            src = edge_np[0].astype(int)
            dst = edge_np[1].astype(int)
            for s, d in zip(src, dst):
                if s < d:
                    nx_g.add_edge(int(s), int(d))
        nx_graphs.append(nx_g)
    return nx_graphs


def collect_train_smiles_molecular(train_graphs, atom_decoder):
    """Convert training molecular graphs to canonical SMILES.

    Parameters
    ----------
    train_graphs : list of Graph
        Training graphs with .x (one-hot), .edge_index, .edge_attr.
    atom_decoder : list of str
        Atom decoder aligned with the dataset preprocessing.

    Returns
    -------
    list of str
        Canonical SMILES for each valid training graph.
    """
    from rdkit_functions import build_molecule_with_partial_charges, mol2smiles

    if atom_decoder is None:
        return None

    smiles_list = []
    for g in train_graphs:
        x_np = g.x if isinstance(g.x, np.ndarray) else tlx.convert_to_numpy(g.x)
        edge_index_np = g.edge_index if isinstance(g.edge_index, np.ndarray) else tlx.convert_to_numpy(g.edge_index)
        ea_np = g.edge_attr if isinstance(g.edge_attr, np.ndarray) else tlx.convert_to_numpy(g.edge_attr)

        n = x_np.shape[0]
        # Dense adjacency with edge types
        adj = np.zeros((n, n), dtype=int)
        if edge_index_np.shape[1] > 0:
            src = edge_index_np[0].astype(int)
            dst = edge_index_np[1].astype(int)
            for idx in range(len(src)):
                s, d = int(src[idx]), int(dst[idx])
                if ea_np is not None and ea_np.ndim == 2 and ea_np.shape[0] == len(src):
                    bond_type = int(np.argmax(ea_np[idx]))
                else:
                    bond_type = 1
                adj[s, d] = bond_type

        atom_types = np.argmax(x_np, axis=-1)
        mol = build_molecule_with_partial_charges(atom_types, adj, atom_decoder)
        smi = mol2smiles(mol)
        if smi is not None:
            smiles_list.append(smi)

    print(f"  Collected {len(smiles_list)} valid SMILES from {len(train_graphs)} training graphs")
    return smiles_list


def evaluate_generated_graphs(generated, dataset_name, graphs, test_ds,
                              dataset_infos, reference_graphs=None,
                              train_graphs=None, cache_dir=None):
    """Full evaluation pipeline for generated graphs.

    For molecular datasets: computes validity, uniqueness, novelty, FCD, etc.
    For synthetic datasets: computes SPECTRE MMD-based graph statistics.

    Parameters
    ----------
    generated : list of (atom_types, edge_types)
        Generated graphs as integer arrays.
    dataset_name : str
        Name of the dataset.
    graphs : list
        Training graphs.
    test_ds : dataset
        Test dataset.
    dataset_infos : dict
        Dataset information dictionary.
    reference_graphs : list, optional
        Reference graphs for FCD computation.
    train_graphs : list, optional
        Training graphs (overrides `graphs` for SMILES collection).
    cache_dir : str, optional
        Directory for caching SMILES.

    Returns
    -------
    dict
        Evaluation metrics.
    """
    fold_metrics = {}
    is_molecular = dataset_name in ('qm9', 'guacamol', 'zinc250k', 'moses')

    if is_molecular:
        atom_decoder = dataset_infos.get('atom_decoder')
        remove_h = dataset_infos.get('remove_h', True)
        train_graph_source = train_graphs if train_graphs is not None else graphs

        # 1. Check in-memory cache
        train_smiles = dataset_infos.get('train_smiles')
        reference_smiles = dataset_infos.get('reference_smiles')

        # 2. Check disk cache if not in memory
        cache_file = None
        ref_cache_file = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"train_smiles_cache_{dataset_name}.pkl")
            ref_cache_file = os.path.join(cache_dir, f"ref_smiles_cache_{dataset_name}.pkl")

        if train_smiles is None and cache_file is not None and os.path.exists(cache_file):
            import pickle
            with open(cache_file, 'rb') as f:
                train_smiles = pickle.load(f)
            dataset_infos['train_smiles'] = train_smiles
            print(f"Loaded {len(train_smiles)} training SMILES from disk cache")

        if reference_smiles is None and ref_cache_file is not None and os.path.exists(ref_cache_file):
            import pickle
            with open(ref_cache_file, 'rb') as f:
                reference_smiles = pickle.load(f)
            dataset_infos['reference_smiles'] = reference_smiles
            print(f"Loaded {len(reference_smiles)} reference SMILES from disk cache")

        if atom_decoder is not None and train_smiles is None:
            print("Collecting training SMILES...")
            train_smiles = collect_train_smiles_molecular(train_graph_source, atom_decoder)
            dataset_infos['train_smiles'] = train_smiles
            if cache_file is not None:
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump(train_smiles, f)
                print(f"Cached training SMILES to disk: {cache_file}")

        if atom_decoder is not None and reference_smiles is None:
            if reference_graphs is not None:
                print("Collecting reference SMILES for FCD...")
                reference_smiles = collect_train_smiles_molecular(reference_graphs, atom_decoder)
            elif test_ds is not None:
                print("Collecting reference SMILES from test_ds...")
                ref_graphs = [test_ds[i] for i in range(len(test_ds))]
                reference_smiles = collect_train_smiles_molecular(ref_graphs, atom_decoder)
            else:
                reference_smiles = train_smiles
            dataset_infos['reference_smiles'] = reference_smiles
            if ref_cache_file is not None:
                import pickle
                with open(ref_cache_file, 'wb') as f:
                    pickle.dump(reference_smiles, f)
                print(f"Cached reference SMILES to disk: {ref_cache_file}")

        if atom_decoder is not None:
            atom_counts = np.zeros(len(atom_decoder), dtype=np.int64)
            max_edge_type = 0
            for atom_types, edge_types in generated:
                atom_types_np = np.asarray(atom_types, dtype=np.int64)
                edge_types_np = np.asarray(edge_types, dtype=np.int64)
                valid_atom_mask = (atom_types_np >= 0) & (atom_types_np < len(atom_decoder))
                if valid_atom_mask.any():
                    atom_counts += np.bincount(atom_types_np[valid_atom_mask], minlength=len(atom_decoder))
                if edge_types_np.size > 0:
                    max_edge_type = max(max_edge_type, int(edge_types_np.max()))

            bond_counts = np.zeros(max_edge_type + 1, dtype=np.int64)
            for _, edge_types in generated:
                edge_types_np = np.asarray(edge_types, dtype=np.int64)
                if edge_types_np.size == 0:
                    continue
                upper = np.triu(edge_types_np, k=1).reshape(-1)
                valid_bonds = upper[upper >= 0]
                if valid_bonds.size > 0:
                    bond_counts += np.bincount(valid_bonds, minlength=len(bond_counts))

            atom_summary = {atom_decoder[i]: int(atom_counts[i]) for i in range(len(atom_decoder))}
            bond_summary = {int(i): int(bond_counts[i]) for i in range(len(bond_counts))}
            print(f"  Generated atom type counts: {atom_summary}")
            print(f"  Generated bond type counts: {bond_summary}")

            print(f"\nEvaluating molecular metrics on {len(generated)} generated graphs...")
            stability_dict, rdkit_metrics, all_smiles, summary = compute_molecular_metrics(
                generated, train_smiles, atom_decoder, remove_h)

            print(f"\n  Stability: {stability_dict}")
            print(f"  Summary: {summary}")
            fold_metrics.update(summary)
            fold_metrics.update(stability_dict)

            try:
                from rdkit_functions import compute_distribution_metrics
                dist_mae = compute_distribution_metrics(
                    generated, dataset_infos, dataset_name)
                fold_metrics.update(dist_mae)
                print(f"  Distribution MAE: {dist_mae}")
            except Exception as e:
                print(f"  Distribution MAE skipped: {e}")

            try:
                from rdkit_functions import compute_fcd
                fcd_score = compute_fcd(all_smiles, reference_smiles)
                fold_metrics['fcd'] = fcd_score
                print(f"  FCD: {fcd_score:.4f}")
            except Exception as e:
                print(f"  FCD skipped: {e}")

            fold_metrics['selection_score'] = compute_selection_score(dataset_name, fold_metrics)
    else:
        from spectre_utils import evaluate_synthetic_graphs

        print("Converting reference graphs to networkx...")
        if reference_graphs is not None:
            reference_nx = graphs_to_networkx(reference_graphs)
        elif test_ds is not None:
            reference_nx = graphs_to_networkx(
                [test_ds[i] for i in range(min(len(test_ds), 200))])
        else:
            reference_nx = graphs_to_networkx(graphs[:min(len(graphs), 200)])

        if train_graphs is not None:
            train_nx = graphs_to_networkx(train_graphs)
        else:
            train_nx = graphs_to_networkx(graphs[:min(len(graphs), 200)])

        metrics = evaluate_synthetic_graphs(
            generated_graphs=generated,
            reference_graphs=reference_nx,
            train_graphs=train_nx,
            dataset_name=dataset_name,
            compute_emd=(dataset_name == 'comm20'),
        )

        alias_pairs = [
            ('sampling/frac_unique', 'frac_unique'),
            ('sampling/frac_non_iso', 'frac_non_iso'),
            ('sampling/frac_unique_non_iso', 'frac_unique_non_iso'),
            ('sampling/frac_unic_non_iso_valid', 'frac_unic_non_iso_valid'),
        ]
        for new_key, old_key in alias_pairs:
            value = metrics.get(old_key)
            if value is not None:
                metrics[new_key] = value

        if dataset_name == 'tree' and 'tree_acc' in metrics:
            metrics.setdefault('valid', metrics['tree_acc'])
        if dataset_name == 'planar' and 'planar_acc' in metrics:
            metrics.setdefault('valid', metrics['planar_acc'])

        if 'valid' in metrics:
            metrics.setdefault('sampling/frac_unic_non_iso_valid', metrics['valid'])
            metrics.setdefault('frac_unic_non_iso_valid', metrics['valid'])

        metrics['selection_score'] = compute_selection_score(dataset_name, metrics)

        fold_metrics.update(metrics)

        print("\n  Evaluation Results:")
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.floating)):
                print(f"    {k}: {float(v):.6f}")
            else:
                print(f"    {k}: {v}")

    if 'selection_score' not in fold_metrics:
        fold_metrics['selection_score'] = compute_selection_score(dataset_name, fold_metrics)

    return fold_metrics
