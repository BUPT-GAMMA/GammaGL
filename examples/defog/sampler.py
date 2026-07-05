"""CTMC sampling loop for DeFoG (Discrete Flow-matching for Graph Generation).

This module contains the core sampling logic extracted from defog_trainer.py,
including the CTMC one-step transition probability computation,
classifier-free guidance helpers, and the main batch sampling loop.
"""

import os
import numpy as np
import tensorlayerx as tlx

from defog_utils import PlaceHolder, apply_node_mask
import torch.nn.functional as F
from flow_matching import sample_discrete_features, sample_discrete_feature_noise
from extra_features import compute_extra_data


def compute_step_probs(R_t_X, R_t_E, X_t, E_t, dt):
    r"""Convert rate matrices to one-step CTMC transition probabilities.

    Matches the original DeFoG logic: zero the current-state column first,
    then write back the stay probability so rows sum to 1.
    """
    step_X = R_t_X * dt
    step_E = R_t_E * dt

    cur_X = tlx.argmax(X_t, axis=-1)
    cur_E = tlx.argmax(E_t, axis=-1)

    # TODO: Performance Optimization
    # Currently converting tensors to numpy for inplace modifications (setting current state
    # to 0 and writing back stay probability). Doing this entirely in pure tensors would
    # prevent CPU-GPU synchronization bottlenecks during sampling.
    step_X_np = tlx.convert_to_numpy(step_X)
    step_E_np = tlx.convert_to_numpy(step_E)
    cur_X_np = tlx.convert_to_numpy(cur_X).astype(np.int64)
    cur_E_np = tlx.convert_to_numpy(cur_E).astype(np.int64)

    bs, n, dx = step_X_np.shape
    _, n1, n2, de = step_E_np.shape

    step_X_np[np.arange(bs)[:, None], np.arange(n)[None, :], cur_X_np] = 0.0
    stay_X = np.clip(1.0 - step_X_np.sum(axis=-1, keepdims=True), a_min=0.0, a_max=None)
    step_X_np[np.arange(bs)[:, None], np.arange(n)[None, :], cur_X_np] = stay_X[..., 0]

    b_idx = np.arange(bs)[:, None, None]
    i_idx = np.arange(n1)[None, :, None]
    j_idx = np.arange(n2)[None, None, :]
    step_E_np[b_idx, i_idx, j_idx, cur_E_np] = 0.0
    stay_E = np.clip(1.0 - step_E_np.sum(axis=-1, keepdims=True), a_min=0.0, a_max=None)
    step_E_np[b_idx, i_idx, j_idx, cur_E_np] = stay_E[..., 0]

    prob_X = tlx.convert_to_tensor(step_X_np.astype(np.float32))
    prob_E = tlx.convert_to_tensor(step_E_np.astype(np.float32))
    return prob_X, prob_E


def _cfg_unconditional_pred(model, X_in, E_in, extra_data, y_t, node_mask):
    r"""Compute unconditional model predictions for classifier-free guidance."""
    y_uncond = tlx.ones_like(y_t) * (-1.0)
    y_in_uncond = tlx.concat([y_uncond, extra_data.y], axis=-1)

    import torch
    with torch.no_grad():
        pred_X_u, pred_E_u, _ = model(X_in, E_in, y_in_uncond, node_mask)
    pred_X_soft_u = tlx.softmax(pred_X_u, axis=-1)
    pred_E_soft_u = tlx.softmax(pred_E_u, axis=-1)
    return pred_X_soft_u, pred_E_soft_u


def sample_batch(model, noise_dist, rate_matrix_designer, time_distorter,
                 extra_features, domain_features, node_dist,
                 sample_steps, batch_size, num_nodes=None,
                 conditional=False, cond_labels=None, guidance_weight=2.0):
    r"""Generate graphs via CTMC sampling.

    Parameters
    ----------
    model : DeFoGModel
        The trained denoiser model.
    noise_dist : NoiseDistribution
        Noise distribution.
    rate_matrix_designer : RateMatrixDesigner
        Rate matrix computer.
    time_distorter : TimeDistorter
        Time distortion for sampling.
    extra_features : callable
        Structural extra features.
    domain_features : callable
        Domain-specific extra features.
    node_dist : ndarray
        Distribution over number of nodes.
    sample_steps : int
        Number of sampling steps.
    batch_size : int
        Number of graphs to generate.
    num_nodes : list, optional
        Pre-specified number of nodes per graph.
    conditional : bool
        Whether to use classifier-free guidance.
    cond_labels : tensor, optional
        Conditional labels ``(batch_size, n_cond)`` for guided generation.
    guidance_weight : float
        Classifier-free guidance weight. Default 2.0.

    Returns
    -------
    list
        List of tuples ``(X_int, E_int)`` for each generated graph.
    """
    model.set_eval()
    import torch
    limit_dist = noise_dist.get_limit_dist()

    # Sample number of nodes
    if num_nodes is None:
        p = node_dist / node_dist.sum()
        n_nodes = np.random.choice(len(node_dist), size=batch_size, p=p)
    else:
        n_nodes = np.array(num_nodes[:batch_size])

    n_max = int(np.max(n_nodes))

    # Build node mask
    node_mask_np = np.zeros((batch_size, n_max), dtype=np.float32)
    for i, n in enumerate(n_nodes):
        node_mask_np[i, :n] = 1.0
    node_mask = tlx.convert_to_tensor(node_mask_np.astype(bool))

    # Sample initial noise from limit distribution
    z = sample_discrete_feature_noise(limit_dist, node_mask)
    X_t = z.X  # (bs, n_max, dx)
    E_t = z.E  # (bs, n_max, n_max, de)

    dx = len(limit_dist.X)
    de = len(limit_dist.E)

    debug_sampling = os.environ.get('DEFOG_DEBUG_SAMPLING', '0') == '1'
    if debug_sampling:
        node_mask_np = tlx.convert_to_numpy(node_mask).astype(bool)
        upper_mask_np = np.triu(np.ones((n_max, n_max), dtype=bool), k=1)[None, :, :]
        valid_edge_mask_np = (
            node_mask_np[:, :, None] & node_mask_np[:, None, :] & upper_mask_np
        )

        def _debug_edge_probs(tag, tensor, step_idx):
            arr = tlx.convert_to_numpy(tensor)
            rows = arr[valid_edge_mask_np]
            if rows.size == 0:
                return
            mean_probs = rows.mean(axis=0)
            print(
                f"edge_probs={np.array2string(mean_probs, precision=4, suppress_small=True)}",
                flush=True,
            )

        def _debug_edge_labels(tag, tensor, step_idx):
            arr = tlx.convert_to_numpy(tensor).astype(np.int64)
            labels = arr[valid_edge_mask_np]
            if labels.size == 0:
                return
            counts = np.bincount(labels, minlength=de)
            print(
                f"edge_counts={counts.tolist()}",
                flush=True,
            )

    for step in range(sample_steps):
        t_int = step
        s_int = step + 1

        t_norm = tlx.convert_to_tensor(
            np.full((batch_size, 1), t_int / sample_steps, dtype=np.float32)
        )
        s_norm = tlx.convert_to_tensor(
            np.full((batch_size, 1), s_int / sample_steps, dtype=np.float32)
        )

        # Avoid failure mode of absorbing transition at t=0
        if noise_dist.transition in ('absorbing', 'absorbfirst') and t_int == 0:
            t_norm = t_norm + 1e-6

        t_dist = time_distorter.sample_ft(t_norm)
        s_dist = time_distorter.sample_ft(s_norm)
        dt = float(tlx.convert_to_numpy(s_dist[0, 0] - t_dist[0, 0]))

        # Build noisy data dict
        if conditional and cond_labels is not None:
            y_t = cond_labels  # (batch_size, n_cond)
        else:
            y_t = tlx.zeros([batch_size, 0], dtype=tlx.float32)

            # TLS half-half conditional sampling: half label=0, half label=1
            if conditional and cond_labels is None:
                half = batch_size // 2
                y_t_np = np.zeros((batch_size, 1), dtype=np.float32)
                y_t_np[half:, 0] = 1.0
                y_t = tlx.convert_to_tensor(y_t_np)
        noisy_data = {
            't': t_dist,
            'X_t': X_t,
            'E_t': E_t,
            'y_t': y_t,
            'node_mask': node_mask,
        }

        # Compute extra features
        extra_data = compute_extra_data(noisy_data, extra_features,
                                         domain_features, noise_dist)

        # Forward pass
        X_in = tlx.concat([X_t, extra_data.X], axis=-1)
        E_in = tlx.concat([E_t, extra_data.E], axis=-1)
        y_in = tlx.concat([y_t, extra_data.y], axis=-1)

        with torch.no_grad():
            pred_X, pred_E, _ = model(X_in, E_in, y_in, node_mask)

        # Softmax predictions
        pred_X_soft = tlx.softmax(pred_X, axis=-1)
        pred_E_soft = tlx.softmax(pred_E, axis=-1)

        if debug_sampling and step < 2:
            _debug_edge_probs('pred_E_soft', pred_E_soft, step)

        is_last_step = (s_int == sample_steps)

        if debug_sampling and step == sample_steps - 1:
            _debug_edge_probs('pred_E_soft', pred_E_soft, step)

        if is_last_step:
            # Final step: sample directly from predictions
            # Apply CFG at prediction level for the final step
            if conditional and cond_labels is not None:
                pred_X_soft_u, pred_E_soft_u = _cfg_unconditional_pred(
                    model, X_in, E_in, extra_data, y_t, node_mask)
                eps_cfg = 1e-6
                w = guidance_weight
                pred_X_soft = tlx.softmax(
                    (1 - w) * tlx.log(pred_X_soft_u + eps_cfg)
                    + w * tlx.log(pred_X_soft + eps_cfg), axis=-1)
                pred_E_soft = tlx.softmax(
                    (1 - w) * tlx.log(pred_E_soft_u + eps_cfg)
                    + w * tlx.log(pred_E_soft + eps_cfg), axis=-1)

            sampled = sample_discrete_features(pred_X_soft, pred_E_soft, node_mask)
            X_t = F.one_hot(sampled.X, dx).float()
            E_t = F.one_hot(sampled.E, de).float()
        else:
            # Compute conditional rate matrix
            R_X, R_E = rate_matrix_designer.compute_graph_rate_matrix(
                t_dist, node_mask, (X_t, E_t), (pred_X_soft, pred_E_soft)
            )

            # Classifier-free guidance: blend rate matrices in log-space
            if conditional and cond_labels is not None:
                pred_X_soft_u, pred_E_soft_u = _cfg_unconditional_pred(
                    model, X_in, E_in, extra_data, y_t, node_mask)
                R_X_u, R_E_u = rate_matrix_designer.compute_graph_rate_matrix(
                    t_dist, node_mask, (X_t, E_t), (pred_X_soft_u, pred_E_soft_u)
                )

                # Log-space geometric interpolation of rate matrices
                eps_cfg = 1e-6
                w = guidance_weight
                R_X = tlx.exp(
                    (1 - w) * tlx.log(R_X_u + eps_cfg)
                    + w * tlx.log(R_X + eps_cfg)
                )
                R_E = tlx.exp(
                    (1 - w) * tlx.log(R_E_u + eps_cfg)
                    + w * tlx.log(R_E + eps_cfg)
                )

            prob_X, prob_E = compute_step_probs(R_X, R_E, X_t, E_t, dt)

            if debug_sampling and step < 2:
                _debug_edge_probs('prob_E', prob_E, step)

            # Match original DeFoG sampling path: sample directly from the
            # CTMC one-step probabilities without extra post-processing.
            sampled = sample_discrete_features(prob_X, prob_E, node_mask)
            if debug_sampling and step < 2:
                _debug_edge_labels('sampled_E', sampled.E, step)
            X_t = F.one_hot(sampled.X, dx).float()
            E_t = F.one_hot(sampled.E, de).float()

        # Mask
        X_t, E_t = apply_node_mask(X_t, E_t, node_mask)

    # Remove virtual classes
    result = noise_dist.ignore_virtual_classes(X_t, E_t)
    X_final, E_final = result[0], result[1]

    # Collapse to integer labels
    X_int = tlx.argmax(X_final, axis=-1)
    E_int = tlx.argmax(E_final, axis=-1)

    # Split into individual graphs
    graphs = []
    for i in range(batch_size):
        n = int(n_nodes[i])
        xi = tlx.convert_to_numpy(X_int[i, :n])
        ei = tlx.convert_to_numpy(E_int[i, :n, :n])
        graphs.append((xi, ei))

    return graphs
