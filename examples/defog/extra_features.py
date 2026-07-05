import numpy as np
import tensorlayerx as tlx
class DenseFeaturePlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y



class DummyExtraFeatures:
    r"""Dummy extra features that return empty tensors."""
    def __call__(self, noisy_data):
        X_t = noisy_data['X_t']
        bs = X_t.shape[0]
        n = X_t.shape[1]
        return DenseFeaturePlaceHolder(
            X=tlx.zeros([bs, n, 0], dtype=tlx.float32),
            E=tlx.zeros([bs, n, n, 0], dtype=tlx.float32),
            y=tlx.zeros([bs, 0], dtype=tlx.float32),
        )


class RRWPFeatures:
    r"""Random Walk Return Probability features.

    Computes ``k`` powers of the (optionally row-normalized) adjacency matrix.

    Parameters
    ----------
    k : int
        Number of random walk steps.
    normalize : bool
        If True, row-normalize the adjacency (random walk matrix).
    """
    def __init__(self, k=10, normalize=True):
        self.k = k
        self.normalize = normalize

    def __call__(self, E, k=None):
        """
        Parameters
        ----------
        E : tensor
            Adjacency ``(bs, n, n)`` (already summed over edge types).
            If E has 4 dims ``(bs, n, n, de)``, the non-edge channels are summed.

        Returns
        -------
        tensor
            RRWP features ``(bs, n, n, k)``.
        """
        if k is None:
            k = self.k

        # Handle both 3-dim (adj) and 4-dim (one-hot edges) inputs
        if len(E.shape) == 4:
            adj = tlx.reduce_sum(E[:, :, :, 1:], axis=-1)  # (bs, n, n)
        else:
            adj = E  # (bs, n, n)

        # Ensure float32 for calculations
        adj = tlx.cast(adj, tlx.float32)

        bs, n = adj.shape[0], adj.shape[1]

        if self.normalize:
            deg = tlx.reduce_sum(adj, axis=-1, keepdims=True)
            deg = tlx.where(deg == 0.0, tlx.ones_like(deg), deg)
            A_norm = adj / deg
        else:
            A_norm = adj

        power = tlx.eye(n, dtype=tlx.float32)
        power = tlx.expand_dims(power, 0)
        power = tlx.tile(power, [bs, 1, 1])

        results = [power]

        for i in range(1, k):
            power = tlx.bmm(power, A_norm)
            results.append(power)

        return tlx.stack(results, axis=-1)


# ============================================================
# K-Node Cycle Counting (matches original DeFoG formulas)
# ============================================================

class KNodeCycles:
    r"""Compute 3, 4, 5, 6-cycle counts per node and graph-level.

    Formulas match the original DeFoG implementation exactly.
    """

    def k_cycles(self, adj):
        """
        Parameters
        ----------
        adj : ndarray
            Adjacency matrix ``(n, n)``.

        Returns
        -------
        tuple
            ``(x_cycles, y_cycles)`` where ``x_cycles`` has shape ``(n, 3)``
            (per-node k3, k4, k5) and ``y_cycles`` has shape ``(4,)``
            (graph-level k3, k4, k5, k6).
        """
        A = adj.astype(np.float64)
        n = A.shape[0]
        d = A.sum(axis=1)  # degree vector

        A2 = A @ A
        A3 = A2 @ A
        A4 = A3 @ A
        A5 = A4 @ A
        A6 = A5 @ A

        # ---- k3 ----
        k3_node = np.diag(A3) / 2.0
        k3_graph = np.trace(A3) / 6.0

        # ---- k4 (matches original: d*(d-1) - (A @ d.unsqueeze(-1)).sum(-1)) ----
        k4_node = (
            np.diag(A4)
            - d * (d - 1)
            - (A @ d[:, np.newaxis]).sum(axis=-1)
        ) / 2.0
        k4_graph = (
            np.trace(A4)
            - np.sum(d * (d - 1))
            - np.sum(A @ d[:, np.newaxis])
        ) / 8.0

        # ---- k5 (matches original: 3 terms) ----
        triangles = np.diag(A3)
        k5_node = (
            np.diag(A5)
            - 2.0 * triangles * d
            - (A @ triangles[:, np.newaxis]).sum(axis=-1)
            + triangles
        ) / 2.0
        k5_graph = (
            np.trace(A5)
            - 2.0 * np.sum(triangles * d)
            - np.sum(A @ triangles[:, np.newaxis])
            + np.sum(triangles)
        ) / 10.0

        # ---- k6 graph only (matches original 10-term formula) ----
        d2 = np.diag(A2)
        a4_diag = np.diag(A4)
        k6_graph = (
            np.trace(A6)
            - 3.0 * np.trace(A3 @ A3)
            + 9.0 * np.sum(A * (A2 ** 2))
            - 6.0 * np.sum(d2 * a4_diag)
            + 6.0 * np.trace(A4)
            - 4.0 * np.trace(A3)
            + 4.0 * np.sum(d2 ** 3 / (d2 + 1e-12) * d2)  # d2^3 summed
            + 3.0 * np.sum(A3)
            - 12.0 * np.sum(d2 ** 2)
            + 4.0 * np.trace(A2)
        ) / 12.0

        # Fix k6 formula to exactly match original batch_trace operations
        term_1 = np.trace(A6)
        term_2 = np.trace(A3 @ A3)
        term_3 = np.sum(A * (A2 ** 2))
        term_4 = np.sum(d2 * a4_diag)
        term_5 = np.trace(A4)
        term_6 = np.trace(A3)
        term_7 = np.sum(d2 ** 3)
        term_8 = np.sum(A3)
        term_9 = np.sum(d2 ** 2)
        term_10 = np.trace(A2)

        k6_graph = (
            term_1
            - 3.0 * term_2
            + 9.0 * term_3
            - 6.0 * term_4
            + 6.0 * term_5
            - 4.0 * term_6
            + 4.0 * term_7
            + 3.0 * term_8
            - 12.0 * term_9
            + 4.0 * term_10
        ) / 12.0

        x_cycles = np.stack([k3_node, k4_node, k5_node], axis=-1).astype(np.float32)
        x_cycles = np.clip(x_cycles, 0, None)

        y_cycles = np.array([k3_graph, k4_graph, k5_graph, k6_graph],
                            dtype=np.float32)
        y_cycles = np.clip(y_cycles, 0, None)

        return x_cycles, y_cycles


class NodeCycleFeatures:
    r"""Compute k-cycle counts per node."""
    def __init__(self):
        self.cycle_counter = KNodeCycles()

    def __call__(self, noisy_data):
        """
        Parameters
        ----------
        noisy_data : dict
            Must contain ``'E_t'`` and ``'node_mask'``.

        Returns
        -------
        tuple
            ``(x_cycles, y_cycles)`` as tensors.
        """
        E = noisy_data['E_t']
        node_mask = noisy_data['node_mask']
        E_np = tlx.convert_to_numpy(E)
        mask_np = tlx.convert_to_numpy(node_mask)
        bs, n, _, de = E_np.shape

        adj = np.sum(E_np[:, :, :, 1:], axis=-1)  # (bs, n, n)

        all_x_cycles = []
        all_y_cycles = []
        for b in range(bs):
            x_cyc, y_cyc = self.cycle_counter.k_cycles(adj[b])
            all_x_cycles.append(x_cyc)
            all_y_cycles.append(y_cyc)

        x_cycles = np.stack(all_x_cycles, axis=0)  # (bs, n, 3) for k3,k4,k5
        y_cycles = np.stack(all_y_cycles, axis=0)  # (bs, 4) for k3,k4,k5,k6

        # Normalize and clamp (matching original)
        x_cycles = x_cycles / 10.0
        y_cycles = y_cycles / 10.0
        x_cycles = np.clip(x_cycles, 0, 1)
        y_cycles = np.clip(y_cycles, 0, 1)

        # Apply node mask to zero out padded nodes (Bug 12 fix)
        x_cycles = x_cycles * mask_np[:, :, np.newaxis]

        return (tlx.convert_to_tensor(x_cycles.astype(np.float32)),
                tlx.convert_to_tensor(y_cycles.astype(np.float32)))


# ============================================================
# EigenFeatures (Laplacian eigenvalue / eigenvector features)
# ============================================================

class EigenFeatures:
    r"""Compute Laplacian eigenvalue and eigenvector features.

    Parameters
    ----------
    mode : str
        ``'eigenvalues'`` for eigenvalue features only.
        ``'all'`` for eigenvalues + eigenvectors + connected component features.
    """
    def __init__(self, mode='eigenvalues'):
        self.mode = mode

    def __call__(self, noisy_data):
        E_t = noisy_data['E_t']
        mask = noisy_data['node_mask']

        E_np = tlx.convert_to_numpy(E_t)
        mask_np = tlx.convert_to_numpy(mask)

        A = np.sum(E_np[:, :, :, 1:], axis=-1).astype(np.float64)  # (bs, n, n)
        bs, n = A.shape[0], A.shape[1]

        # Apply mask to adjacency
        A = A * mask_np[:, np.newaxis, :] * mask_np[:, :, np.newaxis]

        # Compute Laplacian
        L = _compute_laplacian(A, normalize=False)

        # Mask out padding nodes: add large diagonal for non-existent nodes
        eye_n = np.eye(n)[np.newaxis, :, :]  # (1, n, n)
        mask_diag = 2.0 * n * eye_n * (1.0 - mask_np[:, np.newaxis, :]) * (1.0 - mask_np[:, :, np.newaxis])
        L = L * mask_np[:, np.newaxis, :] * mask_np[:, :, np.newaxis] + mask_diag

        if self.mode == 'eigenvalues':
            eigvals = np.linalg.eigvalsh(L)  # (bs, n)
            eigvals = eigvals / (mask_np.sum(axis=1, keepdims=True) + 1e-8)

            n_connected, batch_eigenvalues = _get_eigenvalues_features(eigvals)
            return n_connected, batch_eigenvalues

        elif self.mode == 'all':
            eigvals, eigvectors = np.linalg.eigh(L)
            eigvals = eigvals / (mask_np.sum(axis=1, keepdims=True) + 1e-8)
            eigvectors = eigvectors * mask_np[:, :, np.newaxis] * mask_np[:, np.newaxis, :]

            n_connected, batch_eigenvalues = _get_eigenvalues_features(eigvals)
            nonlcc_indicator, k_lowest_eigvec = _get_eigenvectors_features(
                eigvectors, mask_np, n_connected
            )
            return n_connected, batch_eigenvalues, nonlcc_indicator, k_lowest_eigvec
        else:
            raise ValueError(f"Mode {self.mode} not implemented")


def _compute_laplacian(adjacency, normalize=False):
    """Compute Laplacian from adjacency matrix.

    Parameters
    ----------
    adjacency : ndarray
        ``(bs, n, n)``
    normalize : bool
        If True, compute symmetric normalized Laplacian.

    Returns
    -------
    ndarray
        Laplacian ``(bs, n, n)``.
    """
    diag = adjacency.sum(axis=-1)  # (bs, n)
    n = diag.shape[-1]
    D = np.zeros_like(adjacency)
    for b in range(diag.shape[0]):
        np.fill_diagonal(D[b], diag[b])
    combinatorial = D - adjacency

    if not normalize:
        return (combinatorial + np.transpose(combinatorial, (0, 2, 1))) / 2.0

    diag0 = diag.copy()
    diag[diag == 0] = 1e-12
    diag_norm = 1.0 / np.sqrt(diag)
    D_norm = np.zeros_like(adjacency)
    for b in range(diag.shape[0]):
        np.fill_diagonal(D_norm[b], diag_norm[b])

    eye = np.eye(n)[np.newaxis, :, :]
    L = eye - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + np.transpose(L, (0, 2, 1))) / 2.0


def _get_eigenvalues_features(eigenvalues, k=5):
    """Extract eigenvalue features.

    Parameters
    ----------
    eigenvalues : ndarray
        ``(bs, n)`` eigenvalues.
    k : int
        Number of non-zero eigenvalues to keep.

    Returns
    -------
    tuple
        ``(n_connected_components, first_k_eigenvalues)`` with shapes
        ``(bs, 1)`` and ``(bs, k)``.
    """
    bs, n = eigenvalues.shape
    n_connected_components = (eigenvalues < 1e-5).sum(axis=-1)  # (bs,)
    n_connected_components = np.maximum(n_connected_components, 1)

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = np.hstack(
            [eigenvalues, 2.0 * np.ones((bs, to_extend), dtype=eigenvalues.dtype)]
        )

    indices = np.arange(k)[np.newaxis, :] + n_connected_components[:, np.newaxis]
    first_k_ev = np.take_along_axis(eigenvalues, indices, axis=1)  # (bs, k)

    return (n_connected_components[:, np.newaxis].astype(np.float32),
            first_k_ev.astype(np.float32))


def _get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """Extract eigenvector features.

    Parameters
    ----------
    vectors : ndarray
        Eigenvectors ``(bs, n, n)`` in columns.
    node_mask : ndarray
        ``(bs, n)`` bool.
    n_connected : ndarray
        ``(bs, 1)`` number of connected components.
    k : int
        Number of eigenvectors to keep.

    Returns
    -------
    tuple
        ``(not_lcc_indicator, k_lowest_eigvec)`` with shapes ``(bs, n, 1)``
        and ``(bs, n, k)``.
    """
    bs, n = vectors.shape[0], vectors.shape[1]

    # Create indicator for nodes outside the largest connected component
    first_ev = np.round(vectors[:, :, 0], decimals=3) * node_mask
    random_noise = np.random.randn(bs, n) * (1.0 - node_mask)
    first_ev = first_ev + random_noise

    most_common = np.array([np.median(first_ev[b][node_mask[b].astype(bool)])
                            for b in range(bs)])
    # Use a simpler mode computation
    most_common_list = []
    for b in range(bs):
        vals = first_ev[b][node_mask[b].astype(bool)]
        if len(vals) > 0:
            # Find most common value
            unique_vals, counts = np.unique(np.round(vals, decimals=2), return_counts=True)
            most_common_list.append(unique_vals[counts.argmax()])
        else:
            most_common_list.append(0.0)
    most_common = np.array(most_common_list)

    mask_lcc = ~(np.round(first_ev, decimals=2) == most_common[:, np.newaxis])
    not_lcc_indicator = (mask_lcc * node_mask).astype(np.float32)[:, :, np.newaxis]

    # Get k lowest eigenvectors after connected component eigenvectors
    to_extend = max(int(n_connected.max())) + k - n
    if to_extend > 0:
        vectors = np.concatenate(
            [vectors, np.zeros((bs, n, to_extend), dtype=vectors.dtype)],
            axis=2
        )

    indices = np.arange(k)[np.newaxis, np.newaxis, :] + n_connected[:, :, np.newaxis]
    indices = np.broadcast_to(indices, (bs, n, k)).astype(np.int64)
    first_k_ev = np.take_along_axis(vectors, indices, axis=2)  # (bs, n, k)
    first_k_ev = first_k_ev * node_mask[:, :, np.newaxis]

    return not_lcc_indicator, first_k_ev.astype(np.float32)


# ============================================================
# ExtraFeatures (main dispatcher)
# ============================================================

class ExtraFeatures:
    r"""Compute extra structural features for the DeFoG model.

    Parameters
    ----------
    extra_features_type : str
        Feature type: ``'rrwp'``, ``'rrwp_double'``, ``'rrwp_only'``,
        ``'rrwp_comp'``, ``'cycles'``, ``'eigenvalues'``, ``'all'``,
        or ``None``.
    rrwp_steps : int
        Number of RRWP steps.
    dataset_info : dict
        Dataset information with ``'max_n_nodes'`` key.
    """
    def __init__(self, extra_features_type='rrwp', rrwp_steps=12, dataset_info=None):
        self.features_type = extra_features_type
        self.max_n_nodes = dataset_info.get('max_n_nodes', 100) if dataset_info else 100
        self.rrwp_steps = rrwp_steps

        self.rrwp = RRWPFeatures(k=rrwp_steps, normalize=True)
        self.rwp = RRWPFeatures(k=rrwp_steps, normalize=False)
        self.cycle_features = NodeCycleFeatures()

        if extra_features_type in ('eigenvalues', 'all'):
            self.eigenfeatures = EigenFeatures(mode=extra_features_type)

    def __call__(self, noisy_data):
        """
        Parameters
        ----------
        noisy_data : dict
            Must contain ``'X_t'``, ``'E_t'``, ``'node_mask'``.

        Returns
        -------
        DenseFeaturePlaceHolder
            Extra features for X, E, and y.
        """
        E_t = noisy_data['E_t']
        node_mask = noisy_data['node_mask']
        bs = E_t.shape[0]
        n = E_t.shape[1]

        # n_nodes normalized by max
        n_nodes = tlx.reduce_sum(tlx.cast(node_mask, tlx.float32), axis=1, keepdims=True)
        n_norm = n_nodes / self.max_n_nodes

        if self.features_type is None or self.features_type == 'none':
            return DenseFeaturePlaceHolder(
                X=tlx.zeros([bs, n, 0], dtype=tlx.float32),
                E=tlx.zeros([bs, n, n, 0], dtype=tlx.float32),
                y=tlx.zeros([bs, 0], dtype=tlx.float32),
            )

        if self.features_type == 'cycles':
            x_cycles, y_cycles = self.cycle_features(noisy_data)
            extra_y = tlx.concat([n_norm, y_cycles], axis=-1)
            return DenseFeaturePlaceHolder(
                X=x_cycles,
                E=tlx.zeros([bs, n, n, 0], dtype=tlx.float32),
                y=extra_y,
            )

        if self.features_type == 'eigenvalues':
            eigen_out = self.eigenfeatures(noisy_data)
            n_components, batch_eigenvalues = eigen_out
            x_cycles, y_cycles = self.cycle_features(noisy_data)
            extra_y = tlx.concat([n_norm, y_cycles, n_components, batch_eigenvalues], axis=-1)
            return DenseFeaturePlaceHolder(
                X=x_cycles,
                E=tlx.zeros([bs, n, n, 0], dtype=tlx.float32),
                y=extra_y,
            )

        if self.features_type == 'rrwp':
            # Extract adjacency from one-hot edges: sum of non-"no-edge" channels
            E_adj = _extract_adjacency(E_t)
            rrwp_edge = self.rrwp(E_adj, k=self.rrwp_steps)  # (bs, n, n, k)
            rrwp_node = _extract_diagonal(rrwp_edge)  # (bs, n, k)

            # Initialize eigenfeatures for potential later use
            self.eigenfeatures = EigenFeatures(mode='all')

            # Cycle features for y only (Bug 14 fix: X = rrwp_node only, no cycles)
            x_cycles, y_cycles = self.cycle_features(noisy_data)
            extra_y = tlx.concat([n_norm, y_cycles], axis=-1)

            return DenseFeaturePlaceHolder(
                X=rrwp_node,
                E=rrwp_edge,
                y=extra_y,
            )

        if self.features_type == 'rrwp_double':
            E_adj = _extract_adjacency(E_t)
            rrwp_edge = self.rrwp(E_adj, k=self.rrwp_steps)
            rrwp_edge_wo_norm = self.rwp(E_adj, k=self.rrwp_steps)

            # Normalize unnormalized RRWP
            rrwp_np = tlx.convert_to_numpy(rrwp_edge_wo_norm)
            max_val = rrwp_np.max(axis=(1, 2), keepdims=True)
            max_val = np.where(max_val == 0, 1.0, max_val)
            rrwp_np = rrwp_np / max_val
            rrwp_edge_wo_norm = tlx.convert_to_tensor(rrwp_np)

            rrwp_edge = tlx.concat([rrwp_edge, rrwp_edge_wo_norm], axis=-1)
            rrwp_node = _extract_diagonal(rrwp_edge)

            x_cycles, y_cycles = self.cycle_features(noisy_data)
            extra_y = tlx.concat([n_norm, y_cycles], axis=-1)

            return DenseFeaturePlaceHolder(
                X=rrwp_node,
                E=rrwp_edge,
                y=extra_y,
            )

        if self.features_type == 'rrwp_only':
            E_adj = _extract_adjacency(E_t)
            rrwp_edge = self.rrwp(E_adj, k=self.rrwp_steps)
            rrwp_node = _extract_diagonal(rrwp_edge)

            # No cycle features in y (only n_norm)
            return DenseFeaturePlaceHolder(
                X=rrwp_node,
                E=rrwp_edge,
                y=n_norm,
            )

        if self.features_type == 'rrwp_comp':
            E_adj = _extract_adjacency(E_t)
            half_k = max(1, self.rrwp_steps // 2)
            rrwp_edge = self.rrwp(E_adj, k=half_k)
            rrwp_node = _extract_diagonal(rrwp_edge)

            # Complement adjacency
            comp_adj = 1.0 - _extract_adjacency(E_t)
            comp_rrwp_edge = self.rrwp(comp_adj, k=half_k)
            comp_rrwp_node = _extract_diagonal(comp_rrwp_edge)

            x_cycles, y_cycles = self.cycle_features(noisy_data)
            extra_y = tlx.concat([n_norm, y_cycles], axis=-1)

            return DenseFeaturePlaceHolder(
                X=tlx.concat([rrwp_node, comp_rrwp_node], axis=-1),
                E=tlx.concat([rrwp_edge, comp_rrwp_edge], axis=-1),
                y=extra_y,
            )

        if self.features_type == 'all':
            eigen_out = self.eigenfeatures(noisy_data)
            n_components, batch_eigenvalues, nonlcc_indicator, k_lowest_eigvec = eigen_out

            x_cycles, y_cycles = self.cycle_features(noisy_data)

            # X = cycles + nonlcc_indicator + eigenvectors
            X_feat = tlx.concat([x_cycles,
                                 tlx.convert_to_tensor(nonlcc_indicator),
                                 tlx.convert_to_tensor(k_lowest_eigvec)], axis=-1)

            extra_y = tlx.concat([n_norm, y_cycles,
                                  tlx.convert_to_tensor(n_components),
                                  tlx.convert_to_tensor(batch_eigenvalues)], axis=-1)

            return DenseFeaturePlaceHolder(
                X=X_feat,
                E=tlx.zeros([bs, n, n, 0], dtype=tlx.float32),
                y=extra_y,
            )

        raise ValueError(f"Unknown extra feature type: {self.features_type}")


def _extract_adjacency(E_t):
    """Extract adjacency matrix from one-hot edge features.

    Parameters
    ----------
    E_t : tensor
        Edge features ``(bs, n, n, de)``.

    Returns
    -------
    tensor
        Adjacency ``(bs, n, n)``.
    """
    E_np = tlx.convert_to_numpy(E_t)
    adj = np.sum(E_np[:, :, :, 1:], axis=-1).astype(np.float32)  # (bs, n, n)
    return tlx.convert_to_tensor(adj)


def _extract_diagonal(rrwp_edge):
    """Extract diagonal (node-level) features from RRWP edge features.

    Parameters
    ----------
    rrwp_edge : tensor
        RRWP features ``(bs, n, n, k)``.

    Returns
    -------
    tensor
        Node features ``(bs, n, k)``.
    """
    # TODO: Performance Optimization
    # Converting to numpy and back is inefficient.
    # Can be replaced with advanced tensor indexing or torch.diagonal equivalent.
    rrwp_np = tlx.convert_to_numpy(rrwp_edge)
    bs, n, _, k = rrwp_np.shape
    diag_idx = np.arange(n)
    result = rrwp_np[:, diag_idx, diag_idx, :]  # (bs, n, k)
    return tlx.convert_to_tensor(result)


def compute_extra_data(noisy_data, extra_features, domain_features, noise_dist):
    r"""Compute extra features for the model input.

    Parameters
    ----------
    noisy_data : dict
        Noisy data from ``apply_noise()``.
    extra_features : callable
        Structural feature computer.
    domain_features : callable
        Domain-specific feature computer.
    noise_dist : NoiseDistribution
        Noise distribution (for removing virtual classes before feature computation).

    Returns
    -------
    DenseFeaturePlaceHolder
        Extra features for X, E, y.
    """
    # Strip virtual classes before computing features
    X_t = noisy_data['X_t']
    E_t = noisy_data['E_t']

    noisy_for_features = dict(noisy_data)
    result = noise_dist.ignore_virtual_classes(X_t, E_t)
    noisy_for_features['X_t'] = result[0]
    noisy_for_features['E_t'] = result[1]

    # Compute structural features
    extra = extra_features(noisy_for_features)

    # Compute domain features
    domain = domain_features(noisy_for_features)

    # Concatenate
    extra_X = tlx.concat([extra.X, domain.X], axis=-1) if domain.X.shape[-1] > 0 else extra.X
    extra_E = tlx.concat([extra.E, domain.E], axis=-1) if domain.E.shape[-1] > 0 else extra.E

    # Append timestep to y
    t = noisy_data['t']  # (bs, 1)
    extra_y_parts = []
    if extra.y.shape[-1] > 0:
        extra_y_parts.append(extra.y)
    if domain.y.shape[-1] > 0:
        extra_y_parts.append(domain.y)
    extra_y_parts.append(t)  # timestep always last

    extra_y = tlx.concat(extra_y_parts, axis=-1) if len(extra_y_parts) > 1 else t

    return DenseFeaturePlaceHolder(X=extra_X, E=extra_E, y=extra_y)



class ChargeFeature:
    r"""Compute per-node charge using the original DeFoG molecular feature logic.

    Parameters
    ----------
    remove_h : bool
        Whether hydrogens are removed.
    valencies : list
        Expected valency for each atom type.
    """
    def __init__(self, remove_h=True, valencies=None):
        self.remove_h = remove_h
        if valencies is None:
            valencies = [4, 3, 2, 1]
        self.valencies = np.array(valencies, dtype=np.float32)

    def __call__(self, noisy_data):
        X = tlx.convert_to_numpy(noisy_data['X_t'])
        E = tlx.convert_to_numpy(noisy_data['E_t'])
        dx = X.shape[-1]
        de = E.shape[-1]

        if de == 5:
            bond_orders = np.array([0, 1, 2, 3, 1.5], dtype=np.float32).reshape(1, 1, 1, -1)
        else:
            bond_orders = np.array([0, 1, 2, 3], dtype=np.float32).reshape(1, 1, 1, -1)

        weighted_E = E * bond_orders
        current_valencies = np.argmax(weighted_E, axis=-1).sum(axis=-1).astype(np.float32)

        valencies = self.valencies
        if len(valencies) < dx:
            valencies = np.pad(valencies, (0, dx - len(valencies)))
        valencies = valencies.reshape(1, 1, -1)
        weighted_X = X * valencies
        normal_valencies = np.argmax(weighted_X, axis=-1).astype(np.float32)

        charge = (normal_valencies - current_valencies).astype(np.float32)
        return tlx.convert_to_tensor(charge)


class ValencyFeature:
    r"""Compute per-node valency using the original DeFoG molecular feature logic."""
    def __call__(self, noisy_data):
        E = tlx.convert_to_numpy(noisy_data['E_t'])
        de = E.shape[-1]

        if de == 5:
            bond_orders = np.array([0, 1, 2, 3, 1.5], dtype=np.float32).reshape(1, 1, 1, -1)
        else:
            bond_orders = np.array([0, 1, 2, 3], dtype=np.float32).reshape(1, 1, 1, -1)

        weighted_E = E * bond_orders
        valencies = np.argmax(weighted_E, axis=-1).sum(axis=-1).astype(np.float32)
        return tlx.convert_to_tensor(valencies)


class WeightFeature:
    r"""Compute normalized molecular weight.

    Parameters
    ----------
    max_weight : float
        Maximum molecular weight for normalization.
    atom_weights : list or dict
        Atomic weight for each atom type.
    """
    def __init__(self, max_weight=500.0, atom_weights=None):
        self.max_weight = max_weight
        if atom_weights is None:
            atom_weights = [12.0, 14.0, 16.0, 19.0]
        if isinstance(atom_weights, dict):
            atom_weights = [atom_weights[k] for k in sorted(atom_weights.keys())]
        self.atom_weights = np.array(atom_weights, dtype=np.float32)

    def __call__(self, noisy_data):
        X = tlx.convert_to_numpy(noisy_data['X_t'])
        bs, n, dx = X.shape

        atom_types = np.argmax(X, axis=-1)
        aw = self.atom_weights
        if len(aw) < dx:
            aw = np.pad(aw, (0, dx - len(aw)))

        weights = aw[atom_types]
        mol_weight = np.sum(weights, axis=1, keepdims=True).astype(np.float32)
        mol_weight = mol_weight / self.max_weight
        return tlx.convert_to_tensor(mol_weight)


class ExtraMolecularFeatures:
    r"""Molecular-specific extra features: charge, valency, and weight.

    Parameters
    ----------
    dataset_infos : dict
        Dataset information with keys ``'remove_h'``, ``'valencies'``,
        ``'max_weight'``, ``'atom_weights'``.
    """
    def __init__(self, dataset_infos=None):
        if dataset_infos is None:
            dataset_infos = {}
        self.charge = ChargeFeature(
            remove_h=dataset_infos.get('remove_h', True),
            valencies=dataset_infos.get('valencies', None),
        )
        self.valency = ValencyFeature()
        self.weight = WeightFeature(
            max_weight=dataset_infos.get('max_weight', 500.0),
            atom_weights=dataset_infos.get('atom_weights', None),
        )

    def __call__(self, noisy_data):
        """
        Returns
        -------
        DenseFeaturePlaceHolder
            X: ``(bs, n, 2)`` (charge + valency), E: empty, y: ``(bs, 1)`` (weight).
        """
        charge = self.charge(noisy_data)    # (bs, n, 1)
        valency = self.valency(noisy_data)  # (bs, n, 1)
        weight = self.weight(noisy_data)    # (bs, 1)

        bs = charge.shape[0]
        n = charge.shape[1]

        return DenseFeaturePlaceHolder(
            X=tlx.concat([
                tlx.expand_dims(charge, axis=-1),
                tlx.expand_dims(valency, axis=-1),
            ], axis=-1),
            E=tlx.zeros([bs, n, n, 0], dtype=tlx.float32),
            y=weight,
        )
