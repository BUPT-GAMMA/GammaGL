"""
Synthetic graph evaluation metrics (degree, spectral, clustering, orbit, etc.)
Ported from DeFoG src/analysis/spectre_utils.py for GammaGL.

Key changes from original:
- Removed torch / torch_geometric dependency (pure numpy / networkx)
- Removed wandb dependency (print to stdout)
- Removed graph_tool dependency (SBM accuracy simplified)
- Removed pygsp dependency (wavelet stats disabled by default)
- Uses dist_helper.py from this directory for MMD computation
"""
import os
import copy
import signal
import numpy as np
import networkx as nx
import concurrent.futures
from datetime import datetime
from scipy.linalg import eigvalsh

from dist_helper import compute_mmd, gaussian_emd, gaussian, emd, gaussian_tv

PRINT_TIME = False
__all__ = [
    "degree_stats",
    "clustering_stats",
    "orbit_stats_all",
    "spectral_stats",
    "eval_acc_planar_graph",
    "eval_acc_tree_graph",
]


# ============================================================
# Degree statistics
# ============================================================

def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True, compute_emd=False):
    """MMD between degree distributions of two sets of graphs."""
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for G in graph_ref_list:
            sample_ref.append(np.array(nx.degree_histogram(G)))
        for G in graph_pred_list_remove_empty:
            sample_pred.append(np.array(nx.degree_histogram(G)))

    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


# ============================================================
# Spectral statistics
# ============================================================

def spectral_worker(G, n_eigvals=-1):
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    except Exception:
        eigs = np.zeros(G.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1 : n_eigvals + 1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=True,
                   n_eigvals=-1, compute_emd=False):
    """MMD between spectral distributions of two sets of graphs."""
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                spectral_worker, graph_ref_list, [n_eigvals] * len(graph_ref_list)
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                spectral_worker, graph_pred_list_remove_empty,
                [n_eigvals] * len(graph_pred_list_remove_empty),
            ):
                sample_pred.append(spectral_density)
    else:
        for G in graph_ref_list:
            sample_ref.append(spectral_worker(G, n_eigvals))
        for G in graph_pred_list_remove_empty:
            sample_pred.append(spectral_worker(G, n_eigvals))

    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing spectral mmd: ", elapsed)
    return mmd_dist


# ============================================================
# Clustering statistics
# ============================================================

def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, bins=100,
                     is_parallel=True, compute_emd=False):
    """MMD between clustering coefficient distributions."""
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_ref_list]
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]
            ):
                sample_pred.append(clustering_hist)
    else:
        for G in graph_ref_list:
            clustering_coeffs_list = list(nx.clustering(G).values())
            hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)
        for G in graph_pred_list_remove_empty:
            clustering_coeffs_list = list(nx.clustering(G).values())
            hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

    if compute_emd:
        mmd_dist = compute_mmd(
            sample_ref, sample_pred, kernel=gaussian_emd,
            sigma=1.0 / 10, distance_scaling=bins,
        )
    else:
        mmd_dist = compute_mmd(
            sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10,
        )

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing clustering mmd: ", elapsed)
    return mmd_dist


# ============================================================
# Orbit / Motif statistics (requires compiled orca binary)
# ============================================================

motif_to_indices = {
    "3path": [1, 2],
    "4cycle": [8],
}
COUNT_START_STR = "orbit counts:"


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1
    edges = []
    for u, v in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    """Run the orca binary to compute orbit counts for a graph."""
    from secrets import choice
    from string import ascii_uppercase, digits
    import subprocess as sp

    tmp_fname = 'orca/tmp_{}.txt'.format(''.join(choice(ascii_uppercase + digits) for _ in range(8)))
    tmp_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), tmp_fname)

    f = open(tmp_fname, "w")
    f.write(str(graph.number_of_nodes()) + " " + str(graph.number_of_edges()) + "\n")
    for u, v in edge_list_reindexed(graph):
        f.write(str(u) + " " + str(v) + "\n")
    f.close()

    output = sp.check_output([
        str(os.path.join(os.path.dirname(os.path.realpath(__file__)), "orca/orca")),
        "node", "4", tmp_fname, "std",
    ])
    output = output.decode("utf8").strip()
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR) + 2
    output = output[idx:]
    node_orbit_counts = np.array([
        list(map(int, node_cnts.strip().split(" ")))
        for node_cnts in output.strip("\n").split("\n")
    ])

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def motif_stats(graph_ref_list, graph_pred_list, motif_type="4cycle",
                ground_truth_match=None, bins=100, compute_emd=False):
    total_counts_ref = []
    total_counts_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]
    indices = motif_to_indices[motif_type]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)
        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_ref.append(motif_temp)

    for G in graph_pred_list_remove_empty:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)
        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_pred.append(motif_temp)

    total_counts_ref = np.array(total_counts_ref)[:, None]
    total_counts_pred = np.array(total_counts_pred)[:, None]

    mmd_dist = compute_mmd(
        total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False,
    )
    return mmd_dist


def orbit_stats_all(graph_ref_list, graph_pred_list, compute_emd=False):
    total_counts_ref = []
    total_counts_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)

    if compute_emd:
        mmd_dist = compute_mmd(
            total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False, sigma=30.0,
        )
    else:
        mmd_dist = compute_mmd(
            total_counts_ref, total_counts_pred, kernel=gaussian_tv, is_hist=False, sigma=30.0,
        )
    return mmd_dist


# ============================================================
# Graph accuracy checks
# ============================================================

def eval_acc_planar_graph(G_list):
    count = 0
    for gg in G_list:
        if nx.is_connected(gg) and nx.check_planarity(gg)[0]:
            count += 1
    return count / float(len(G_list))


def eval_acc_tree_graph(G_list):
    count = 0
    for gg in G_list:
        if nx.is_tree(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_lobster_graph(G_list):
    count = 0
    for gg in G_list:
        if is_lobster_graph(gg):
            count += 1
    return count / float(len(G_list))


def is_lobster_graph(G):
    if nx.is_tree(G):
        G = G.copy()
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)
        num_nodes = len(G.nodes())
        num_degree_one = [d for n, d in G.degree() if d == 1]
        num_degree_two = [d for n, d in G.degree() if d == 2]
        if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
            return True
        elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
            return True
        else:
            return False
    else:
        return False


# ============================================================
# Uniqueness / isomorphism fractions
# ============================================================

def eval_fraction_unique(fake_graphs, precise=False):
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True
        if not fake_g.number_of_nodes() == 0:
            for fake_old in fake_evaluated:
                if precise:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.is_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
                else:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.could_be_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
        if unique:
            fake_evaluated.append(fake_g)
    return (float(len(fake_graphs)) - count_non_unique) / float(len(fake_graphs))


def eval_fraction_isomorphic(fake_graphs, train_graphs):
    count = 0
    for fake_g in fake_graphs:
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(fake_g, train_g):
                if nx.is_isomorphic(fake_g, train_g):
                    count += 1
                    break
    return count / float(len(fake_graphs))


def eval_fraction_unique_non_isomorphic_valid(fake_graphs, train_graphs, validity_func):
    count_non_unique = 0
    count_isomorphic = 0
    count_valid = 0
    fake_evaluated = []

    for fake_g in fake_graphs:
        unique = True
        if fake_g.number_of_nodes() != 0:
            for fake_old in fake_evaluated:
                if nx.faster_could_be_isomorphic(fake_g, fake_old):
                    if nx.is_isomorphic(fake_g, fake_old):
                        count_non_unique += 1
                        unique = False
                        break
        if unique:
            fake_evaluated.append(fake_g)
            non_isomorphic = True
            for train_g in train_graphs:
                if nx.faster_could_be_isomorphic(fake_g, train_g):
                    if nx.is_isomorphic(fake_g, train_g):
                        count_isomorphic += 1
                        non_isomorphic = False
                        break
            if non_isomorphic and validity_func is not None and validity_func(fake_g):
                count_valid += 1

    total = float(len(fake_graphs))
    frac_unique = (total - count_non_unique) / total
    frac_unique_non_isomorphic = (total - count_non_unique - count_isomorphic) / total
    frac_unique_non_isomorphic_valid = count_valid / total
    return frac_unique, frac_unique_non_isomorphic, frac_unique_non_isomorphic_valid


# ============================================================
# Top-level evaluation function
# ============================================================

def evaluate_synthetic_graphs(generated_graphs, reference_graphs, train_graphs,
                               dataset_name, compute_emd=False):
    """Evaluate generated synthetic graphs against reference/test graphs.

    Parameters
    ----------
    generated_graphs : list of (X_int, E_int) tuples
        Generated graphs from the model.
    reference_graphs : list of nx.Graph
        Reference (test/val) graphs as networkx objects.
    train_graphs : list of nx.Graph
        Training graphs (for novelty computation).
    dataset_name : str
        One of 'planar', 'tree', 'sbm'.
    compute_emd : bool
        Whether to use EMD-based MMD (slower but more accurate).

    Returns
    -------
    dict
        Evaluation metrics.
    """
    # Convert generated (X_int, E_int) tuples to networkx graphs
    networkx_graphs = []
    for graph in generated_graphs:
        node_types, edge_types = graph
        A = (edge_types > 0).astype(int)
        nx_graph = nx.from_numpy_array(A)
        networkx_graphs.append(nx_graph)

    print(f"\nEvaluating {len(networkx_graphs)} generated graphs "
          f"against {len(reference_graphs)} reference graphs "
          f"(dataset: {dataset_name})")

    metrics = {}

    # --- Degree MMD ---
    print("  Computing degree stats...")
    metrics['degree'] = degree_stats(
        reference_graphs, networkx_graphs, is_parallel=True, compute_emd=compute_emd)
    print(f"    Degree MMD: {metrics['degree']:.6f}")

    # --- Spectral MMD ---
    print("  Computing spectral stats...")
    metrics['spectre'] = spectral_stats(
        reference_graphs, networkx_graphs, is_parallel=True, compute_emd=compute_emd)
    print(f"    Spectral MMD: {metrics['spectre']:.6f}")

    # --- Clustering MMD ---
    print("  Computing clustering stats...")
    metrics['clustering'] = clustering_stats(
        reference_graphs, networkx_graphs, bins=100, is_parallel=True, compute_emd=compute_emd)
    print(f"    Clustering MMD: {metrics['clustering']:.6f}")

    # --- Orbit MMD (requires orca binary) ---
    try:
        print("  Computing orbit stats...")
        metrics['orbit'] = orbit_stats_all(
            reference_graphs, networkx_graphs, compute_emd=compute_emd)
        print(f"    Orbit MMD: {metrics['orbit']:.6f}")
    except Exception as e:
        print(f"    Orbit stats skipped (orca not available): {e}")

    # --- Wavelet / Spectral Filter MMD ---
    try:
        print("  Computing wavelet stats...")
        metrics['wavelet'] = spectral_filter_stats(
            reference_graphs, networkx_graphs,
            n_filters=12, is_parallel=True, compute_emd=compute_emd)
        print(f"    Wavelet MMD: {metrics['wavelet']:.6f}")
    except Exception as e:
        print(f"    Wavelet stats skipped: {e}")

    # --- Validity (planar / tree / SBM accuracy) ---
    if dataset_name == 'planar':
        print("  Computing planar accuracy...")
        metrics['planar_acc'] = eval_acc_planar_graph(networkx_graphs)
        print(f"    Planar accuracy: {metrics['planar_acc']:.4f}")
        validity_func = lambda g: nx.is_connected(g) and nx.check_planarity(g)[0]
    elif dataset_name == 'tree':
        print("  Computing tree accuracy...")
        metrics['tree_acc'] = eval_acc_tree_graph(networkx_graphs)
        print(f"    Tree accuracy: {metrics['tree_acc']:.4f}")
        validity_func = nx.is_tree
    elif dataset_name == 'sbm':
        print("  SBM accuracy skipped (requires graph_tool)")
        validity_func = None
    elif dataset_name == 'comm20':
        print("  Comm20 accuracy skipped (no dedicated checker)")
        validity_func = nx.is_connected
    else:
        validity_func = None

    # --- Uniqueness / Novelty ---
    if train_graphs is not None and len(train_graphs) > 0:
        print("  Computing uniqueness and novelty...")
        frac_unique = eval_fraction_unique(networkx_graphs)
        frac_non_iso = 1.0 - eval_fraction_isomorphic(networkx_graphs, train_graphs)
        metrics['frac_unique'] = frac_unique
        metrics['frac_non_iso'] = frac_non_iso
        print(f"    Unique: {frac_unique:.4f}, Non-isomorphic: {frac_non_iso:.4f}")

        if validity_func is not None:
            valid_count = sum(1 for g in networkx_graphs if validity_func(g))
            metrics['valid'] = valid_count / len(networkx_graphs)
            (
                frac_unique,
                frac_unique_non_iso,
                frac_unic_non_iso_valid,
            ) = eval_fraction_unique_non_isomorphic_valid(
                networkx_graphs,
                train_graphs,
                validity_func,
            )
            metrics['frac_unique'] = frac_unique
            metrics['frac_unique_non_iso'] = frac_unique_non_iso
            metrics['frac_unic_non_iso_valid'] = frac_unic_non_iso_valid
            print(
                f"    Valid: {metrics['valid']:.4f}, "
                f"Unique non-iso: {frac_unique_non_iso:.4f}, "
                f"Unique non-iso valid: {frac_unic_non_iso_valid:.4f}"
            )

    return metrics


# ============================================================
# Wavelet / Spectral Filter statistics (pure numpy, no pygsp)
# ============================================================

def eigh_worker(G):
    """Compute eigenvalues and eigenvectors of normalized Laplacian."""
    try:
        L = nx.normalized_laplacian_matrix(G).todense()
        eigvals, eigvecs = np.linalg.eigh(np.asarray(L))
    except Exception:
        n = G.number_of_nodes()
        eigvals = np.zeros(n)
        eigvecs = np.zeros((n, n))
    return eigvals, eigvecs


def compute_list_eigh(graph_list, is_parallel=True):
    """Compute eigendecomposition for a list of graphs."""
    eigval_list = []
    eigvec_list = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for eigvals, eigvecs in executor.map(eigh_worker, graph_list):
                eigval_list.append(eigvals)
                eigvec_list.append(eigvecs)
    else:
        for G in graph_list:
            eigvals, eigvecs = eigh_worker(G)
            eigval_list.append(eigvals)
            eigvec_list.append(eigvecs)
    return eigval_list, eigvec_list


def _heat_kernel_wavelets(eigvals, eigvecs, scales):
    """Compute heat kernel wavelet responses.

    Parameters
    ----------
    eigvals : ndarray
        Eigenvalues ``(n,)``.
    eigvecs : ndarray
        Eigenvectors ``(n, n)``.
    scales : list of float
        Wavelet scales.

    Returns
    -------
    ndarray
        Wavelet responses ``(n_scales, n, n)``.
    """
    n = len(eigvals)
    results = []
    for s in scales:
        # Heat kernel: h(s, lambda) = exp(-s * lambda)
        h = np.exp(-s * eigvals)  # (n,)
        # Wavelet = U @ diag(h) @ U^T
        W = eigvecs @ np.diag(h) @ eigvecs.T  # (n, n)
        results.append(W)
    return np.array(results)  # (n_scales, n, n)


def _get_spectral_filter_worker_np(eigvals, eigvecs, n_filters=12, bound=1.4):
    """Pure-numpy spectral filter response (replaces pygsp-based version).

    Uses heat kernel wavelets at logarithmically spaced scales.
    """
    scales = np.logspace(-2, 1, n_filters)  # 0.01 to 10
    wavelets = _heat_kernel_wavelets(eigvals, eigvecs, scales)  # (nf, n, n)

    # Compute squared norm per node per filter
    norm_filt = np.sum(wavelets ** 2, axis=2)  # (nf, n)

    # Histogram per filter
    hist = np.array([
        np.histogram(norm_filt[i], range=[0, bound], bins=100)[0]
        for i in range(n_filters)
    ])
    return hist.flatten()


def spectral_filter_stats(graph_ref_list, graph_pred_list,
                          n_filters=12, is_parallel=True, compute_emd=False):
    """Compute MMD between spectral filter wavelet responses.

    Uses heat kernel wavelets as a drop-in replacement for pygsp Abspline
    filters (no pygsp dependency required).

    Parameters
    ----------
    graph_ref_list, graph_pred_list : list of nx.Graph
    n_filters : int
        Number of wavelet scales.
    is_parallel : bool
    compute_emd : bool
    """
    print("  Computing eigendecompositions for wavelet stats...")
    eigval_ref, eigvec_ref = compute_list_eigh(graph_ref_list, is_parallel)
    eigval_pred, eigvec_pred = compute_list_eigh(
        [G for G in graph_pred_list if G.number_of_nodes() > 0],
        is_parallel
    )

    bound = 1.4
    sample_ref = []
    sample_pred = []

    prev = datetime.now()
    for i in range(len(eigval_ref)):
        sample_ref.append(_get_spectral_filter_worker_np(
            eigval_ref[i], eigvec_ref[i], n_filters, bound))
    for i in range(len(eigval_pred)):
        sample_pred.append(_get_spectral_filter_worker_np(
            eigval_pred[i], eigvec_pred[i], n_filters, bound))

    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing wavelet mmd: ", elapsed)
    return mmd_dist
