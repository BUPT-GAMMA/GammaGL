import numpy as np
import tensorlayerx as tlx
import networkx as nx

def compute_tls_metrics(generated, cond_labels, train_graphs):
    from gammagl.datasets.tls_dataset import CellGraph, PHENOTYPE_DECODER

    train_hashes = set()
    if train_graphs is not None:
        for g in train_graphs:
            x_np = g.x if isinstance(g.x, np.ndarray) else tlx.convert_to_numpy(g.x)
            edge_np = g.edge_index if isinstance(g.edge_index, np.ndarray) else tlx.convert_to_numpy(g.edge_index)
            nx_g = nx.Graph()
            n = x_np.shape[0]
            nx_g.add_nodes_from(range(n))
            if x_np.shape[-1] == 9:
                node_types = np.argmax(x_np, axis=-1)
                for i in range(n):
                    nx_g.nodes[i]['phenotype'] = PHENOTYPE_DECODER[node_types[i]]
            if edge_np.shape[1] > 0:
                src, dst = edge_np[0].astype(int), edge_np[1].astype(int)
                for s, d in zip(src, dst):
                    if s < d:
                        nx_g.add_edge(s, d)
            train_hashes.add(nx.weisfeiler_lehman_graph_hash(nx_g, node_attr='phenotype'))

    valid_graphs = []
    tls_correct = 0
    total_cond = 0
    generated_hashes = []

    for idx, (atom_types, edge_types) in enumerate(generated):
        atom_types_np = np.asarray(atom_types, dtype=np.int64)
        edge_types_np = np.asarray(edge_types, dtype=np.int64)
        n = atom_types_np.shape[0]
        nx_g = nx.Graph()

        for i in range(n):
            node_type_idx = int(atom_types_np[i])
            phenotype = PHENOTYPE_DECODER[node_type_idx] if node_type_idx < len(PHENOTYPE_DECODER) else 'Marker'
            nx_g.add_node(i, phenotype=phenotype)

        if edge_types_np.size > 0:
            upper = np.triu(edge_types_np, k=1)
            src, dst = np.nonzero(upper)
            for s, d in zip(src, dst):
                if upper[s, d] > 0:
                    nx_g.add_edge(int(s), int(d))

        is_valid = True
        try:
            if not nx.is_connected(nx_g):
                is_valid = False
            elif not nx.check_planarity(nx_g)[0]:
                is_valid = False
        except Exception:
            is_valid = False

        if is_valid:
            valid_graphs.append(nx_g)
            generated_hashes.append(nx.weisfeiler_lehman_graph_hash(nx_g, node_attr='phenotype'))

            if cond_labels is not None and idx < len(cond_labels):
                label_tensor = cond_labels[idx]
                try:
                    label = int(tlx.convert_to_numpy(label_tensor).item())
                    cell_g = CellGraph(nx_g)
                    if label == 1 and cell_g.has_high_TLS():
                        tls_correct += 1
                    elif label == 0 and cell_g.has_low_TLS():
                        tls_correct += 1
                    total_cond += 1
                except Exception:
                    pass

    num_generated = len(generated)
    num_valid = len(valid_graphs)
    valid_ratio = num_valid / num_generated if num_generated > 0 else 0.0

    unique_hashes = set(generated_hashes)
    num_unique = len(unique_hashes)
    unique_ratio = num_unique / num_valid if num_valid > 0 else 0.0

    novel_hashes = unique_hashes - train_hashes
    num_novel = len(novel_hashes)
    novel_ratio = num_novel / num_unique if num_unique > 0 else 0.0

    v_u_n = valid_ratio * unique_ratio * novel_ratio

    metrics = {
        'valid': valid_ratio,
        'unique': unique_ratio,
        'novel': novel_ratio,
        'V.U.N.': v_u_n,
    }

    if total_cond > 0:
        metrics['tls_validity'] = tls_correct / total_cond

    return metrics
