###############################################################################
#
# Adapted from https://github.com/lrjconan/GRAN/ which in turn is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
import numpy as np
import concurrent.futures
from functools import partial

try:
    import pyemd
    from scipy.linalg import toeplitz
    SPECTRE_METRICS_AVAILABLE = True
except ImportError:
    SPECTRE_METRICS_AVAILABLE = False
    import warnings
    warnings.warn("pyemd or scipy not found, spectre evaluation metrics will fail.")





def emd(x, y, sigma=1.0, distance_scaling=1.0):
    """EMD
    Args:
        x, y: 1D pmf of two distributions with the same support
        sigma: standard deviation
    """
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(float)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    return np.abs(pyemd.emd(x, y, distance_mat))


def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    """Gaussian kernel with squared distance in exponential term replaced by EMD
    Args:
        x, y: 1D pmf of two distributions with the same support
        sigma: standard deviation
    """
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(float)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = pyemd.emd(x, y, distance_mat)
    return np.exp(-emd * emd / (2 * sigma * sigma))


def gaussian(x, y, sigma=1.0):
    support_size = max(len(x), len(y))
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def gaussian_tv(x, y, sigma=1.0):
    support_size = max(len(x), len(y))
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))


def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d


def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)


def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    """Discrepancy between 2 samples"""
    d = 0

    if not is_parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for dist in executor.map(
                kernel_parallel_worker,
                [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1],
            ):
                d += dist
    if len(samples1) * len(samples2) > 0:
        d /= len(samples1) * len(samples2)
    else:
        d = 1e6
    return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """MMD between two samples"""
    # normalize histograms into pmf
    if is_hist:
        samples1 = [s1 / (np.sum(s1) + 1e-6) for s1 in samples1]
        samples2 = [s2 / (np.sum(s2) + 1e-6) for s2 in samples2]
    mmd = (
        disc(samples1, samples1, kernel, *args, **kwargs)
        + disc(samples2, samples2, kernel, *args, **kwargs)
        - 2 * disc(samples1, samples2, kernel, *args, **kwargs)
    )

    mmd = np.abs(mmd)



    return mmd





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
import concurrent.futures
from datetime import datetime

try:
    import networkx as nx
    from scipy.linalg import eigvalsh
except ImportError:
    pass


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
        print("  SBM accuracy using official DiGress/DeFoG graph_tool MDL with Wald Test...")
        def is_sbm_valid_official(g, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000):
            import json
            import subprocess
            import os
            
            # Extract edges from networkx graph to pass to subprocess
            edges = list(g.edges())
            data = {
                'edges': edges,
                'p_intra': p_intra,
                'p_inter': p_inter,
                'strict': strict,
                'refinement_steps': refinement_steps
            }
            
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sbm_isolated.py")
            try:
                # Run the isolated script
                result = subprocess.run(
                    ["python", script_path],
                    input=json.dumps(data).encode('utf-8'),
                    capture_output=True,
                    check=True
                )
                output = result.stdout.decode('utf-8').strip()
                return output == "True"
            except subprocess.CalledProcessError as e:
                # If there's a serious error in the script (like missing graph-tool)
                print(f"SBM Evaluation error: {e.stderr.decode('utf-8')}", file=sys.stderr)
                return False
        
        validity_func = is_sbm_valid_official
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
