import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TL_BACKEND'] = 'torch'

import sys
import argparse
import random
import numpy as np
import torch
import threading
import copy
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, CURRENT_DIR)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

from gammagl.models.defog import DeFoGModel
from gammagl.data import Graph
from gammagl.loader import DataLoader

from defog_utils import (PlaceHolder, to_dense,
                         backend_one_hot, EMA)
from flow_utils import p_xt_g_x1
from flow_matching_utils import (sample_discrete_features,
                                  sample_discrete_feature_noise)
from noise_distribution import NoiseDistribution
from rate_matrix import RateMatrixDesigner
from time_distorter import TimeDistorter
from extra_features import ExtraFeatures
from extra_features_molecular import ExtraMolecularFeatures, DummyMolecularFeatures
from train_metrics import TrainLossDiscrete
from rdkit_functions import compute_molecular_metrics


# ============================================================
# Original DeFoG-aligned presets / checkpoint helpers
# ============================================================

_BASE_EXPERIMENT_PRESET = {
    'transition': 'marginal',
    'extra_features': 'rrwp',
    'rrwp_steps': 12,
    'n_layers': 5,
    'hidden_mlp_X': 256,
    'hidden_mlp_E': 128,
    'hidden_mlp_y': 128,
    'dx': 256,
    'de': 64,
    'dy': 64,
    'n_head': 8,
    'dim_ffX': 256,
    'dim_ffE': 128,
    'dim_ffy': 128,
    'n_epochs': 1000,
    'batch_size': 512,
    'lr': 2e-4,
    'weight_decay': 1e-12,
    'ema_decay': 0.0,
    'train_distortion': 'identity',
    'sample_steps': 1000,
    'sample_distortion': 'identity',
    'eta': 0.0,
    'omega': 0.0,
    'rdb': 'general',
    'rdb_crit': 'max_marginal',
    'num_sample_fold': 1,
    'sample_every_val': 4,
    'check_val_every_n_epochs': 5,
    'val_num_samples': 512,
}

_DATASET_PRESETS = {
    'planar': {
        'n_layers': 10,
        'hidden_mlp_X': 128,
        'hidden_mlp_E': 64,
        'hidden_mlp_y': 128,
        'dim_ffE': 64,
        'dim_ffy': 256,
        'n_epochs': 100000,
        'batch_size': 64,
        'sample_distortion': 'polydec',
        'omega': 0.05,
        'eta': 50.0,
        'sample_every_val': 1,
        'check_val_every_n_epochs': 2000,
        'val_num_samples': 40,
    },
    'tree': {
        'n_layers': 10,
        'hidden_mlp_X': 128,
        'hidden_mlp_E': 64,
        'hidden_mlp_y': 128,
        'dim_ffE': 64,
        'dim_ffy': 256,
        'n_epochs': 100000,
        'batch_size': 64,
        'train_distortion': 'polydec',
        'sample_distortion': 'polydec',
        'sample_every_val': 1,
        'check_val_every_n_epochs': 2000,
        'val_num_samples': 40,
    },
    'sbm': {
        'transition': 'absorbfirst',
        'rrwp_steps': 20,
        'n_layers': 8,
        'hidden_mlp_X': 128,
        'hidden_mlp_E': 64,
        'hidden_mlp_y': 128,
        'dim_ffE': 64,
        'dim_ffy': 256,
        'n_epochs': 50000,
        'batch_size': 32,
        'sample_every_val': 1,
        'check_val_every_n_epochs': 2000,
        'val_num_samples': 40,
    },
    'qm9': {
        'n_layers': 9,
        'n_epochs': 1000,
        'batch_size': 1024,
        'sample_steps': 500,
        'sample_distortion': 'polydec',
        'sample_every_val': 1,
        'check_val_every_n_epochs': 50,
        'val_num_samples': 512,
    },
    'zinc250k': {
        'rrwp_steps': 20,
        'n_layers': 12,
        'hidden_mlp_X': 256,
        'hidden_mlp_E': 128,
        'hidden_mlp_y': 256,
        'dy': 128,
        'dim_ffX': 256,
        'dim_ffE': 128,
        'dim_ffy': 256,
        'n_epochs': 300,
        'batch_size': 256,
        'train_distortion': 'polydec',
        'sample_distortion': 'polydec',
        'omega': 0.1,
        'eta': 300.0,
        'sample_every_val': 2,
        'check_val_every_n_epochs': 4,
        'val_num_samples': 256,
    },
    'guacamol': {
        'rrwp_steps': 20,
        'n_layers': 12,
        'hidden_mlp_X': 256,
        'hidden_mlp_E': 128,
        'hidden_mlp_y': 256,
        'dy': 128,
        'dim_ffX': 256,
        'dim_ffE': 128,
        'dim_ffy': 256,
        'n_epochs': 1000,
        'batch_size': 64,
        'train_distortion': 'polydec',
        'sample_distortion': 'polydec',
        'omega': 0.1,
        'eta': 300.0,
        'sample_every_val': 2,
        'check_val_every_n_epochs': 2,
        'val_num_samples': 500,
    },
    'moses': {
        'rrwp_steps': 20,
        'n_layers': 12,
        'hidden_mlp_X': 256,
        'hidden_mlp_E': 128,
        'hidden_mlp_y': 256,
        'dy': 128,
        'dim_ffX': 256,
        'dim_ffE': 128,
        'dim_ffy': 256,
        'n_epochs': 300,
        'batch_size': 256,
        'train_distortion': 'polydec',
        'sample_distortion': 'polydec',
        'omega': 0.5,
        'eta': 200.0,
        'sample_every_val': 4,
        'check_val_every_n_epochs': 1,
        'val_num_samples': 256,
    },
    'tls': {
        'n_layers': 10,
        'rrwp_steps': 20,
        'hidden_mlp_X': 128,
        'hidden_mlp_E': 64,
        'hidden_mlp_y': 128,
        'dim_ffE': 64,
        'dim_ffy': 256,
        'n_epochs': 100000,
        'batch_size': 64,
        'sample_distortion': 'polydec',
        'omega': 0.05,
        'eta': 0.0,
        'sample_every_val': 1,
        'check_val_every_n_epochs': 2000,
        'val_num_samples': 40,
    },
    'comm20': {
        'n_layers': 8,
        'n_epochs': 1000000,
        'batch_size': 256,
        'sample_every_val': 10,
        'check_val_every_n_epochs': 1000,
        'val_num_samples': 20,
    },
}


def _get_explicit_cli_dests(parser, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    explicit = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt in argv or any(arg.startswith(opt + '=') for arg in argv):
                explicit.add(action.dest)
                break
    return explicit


def apply_dataset_preset(args, parser, argv=None):
    dataset = getattr(args, 'dataset', None)
    if dataset in (None, 'synthetic'):
        return args

    preset = dict(_BASE_EXPERIMENT_PRESET)
    preset.update(_DATASET_PRESETS.get(dataset, {}))
    explicit = _get_explicit_cli_dests(parser, argv)
    applied = []

    for key, value in preset.items():
        if hasattr(args, key) and key not in explicit:
            setattr(args, key, value)
            applied.append(key)

    if applied:
        preview = ', '.join(applied[:8])
        suffix = ' ...' if len(applied) > 8 else ''
        print(f"Applied original DeFoG preset for {dataset}: {preview}{suffix}")

    return args


def save_model_snapshot(model, ema, save_dir, prefix, output_dims=None):
    model_path = os.path.join(save_dir, f'{prefix}_model.npz')
    ema_path = os.path.join(save_dir, f'{prefix}_ema.pkl')
    model.save_weights(model_path, format='npz_dict')

    if ema is not None:
        with open(ema_path, 'wb') as f:
            f.write(ema.state_dict())
    elif os.path.exists(ema_path):
        os.remove(ema_path)

    # Save output_dims for reproducibility when loading for sampling
    if output_dims is not None:
        import json as _json
        config_path = os.path.join(save_dir, 'model_config.json')
        with open(config_path, 'w') as _f:
            _json.dump({'output_dims': output_dims}, _f, indent=2)

    return model_path, ema_path


def load_model_snapshot_for_sampling(model, save_dir, ema_decay=0.0):
    prefixes = ['best', 'last']
    chosen_prefix = None
    model_path = None

    for prefix in prefixes:
        candidate = os.path.join(save_dir, f'{prefix}_model.npz')
        if os.path.exists(candidate):
            chosen_prefix = prefix
            model_path = candidate
            break

    if model_path is None:
        raise FileNotFoundError(
            f"No sampling checkpoint found in {save_dir}. Expected best_model.npz or last_model.npz"
        )

    model.load_weights(model_path, format='npz_dict')
    print(f"Loaded model from {model_path}")

    ema = None
    ema_path = os.path.join(save_dir, f'{chosen_prefix}_ema.pkl')
    if os.path.exists(ema_path):
        ema = EMA(model, decay=max(float(ema_decay), 0.999))
        with open(ema_path, 'rb') as f:
            ema.load_state_dict(f.read())
        ema.swap_in(model)
        print(f"  Using EMA weights from {ema_path}")

    return model_path, ema


def compute_selection_score(dataset_name, metrics):
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


# ============================================================
# AdamW Optimizer Wrapper for TensorLayerX (torch backend)
# ============================================================

class AdamW(tlx.optimizers.Adam):
    r"""AdamW optimizer that uses ``torch.optim.AdamW`` (decoupled weight decay).

    Drop-in replacement for ``tlx.optimizers.Adam`` with proper AdamW semantics
    and ``amsgrad`` support.
    """

    def __init__(self, lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, grad_clip=None):
        self.amsgrad = amsgrad
        super().__init__(lr=lr, beta_1=beta_1, beta_2=beta_2, eps=eps,
                         weight_decay=weight_decay, grad_clip=grad_clip)

    def gradient(self, loss, weights=None, return_grad=True):
        import torch
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_adam = torch.optim.AdamW(
                params=weights,
                lr=self.lr,
                betas=(self.beta_1, self.beta_2),
                eps=self.eps,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad,
            )
            self.init_optim = True
        self.optimizer_adam.zero_grad()

        if not torch.isfinite(loss):
            print("[warn:optim] Non-finite loss before backward; skipping optimizer step", flush=True)
            if return_grad:
                return [torch.zeros_like(w) for w in weights]
            return None

        loss.backward()

        nonfinite_grad_tensors = 0
        for w in weights:
            grad = w.grad
            if grad is None:
                continue
            if not torch.isfinite(grad).all():
                nonfinite_grad_tensors += 1
                grad.data = torch.nan_to_num(grad.data, nan=0.0, posinf=0.0, neginf=0.0)

        if nonfinite_grad_tensors > 0:
            print(
                f"[warn:optim] Sanitized non-finite gradients in {nonfinite_grad_tensors} tensors",
                flush=True,
            )

        if self.grad_clip is not None:
            grad_norm = self.grad_clip(weights)
            if isinstance(grad_norm, torch.Tensor) and not torch.isfinite(grad_norm):
                print("[warn:optim] Non-finite grad norm after clipping; zeroing gradients", flush=True)
                for w in weights:
                    if w.grad is not None:
                        w.grad.zero_()

        if return_grad:
            return [w.grad for w in weights]
        else:
            return None

    def apply_gradients(self, grads_and_vars=None, closure=None):
        if not self.init_optim:
            raise AttributeError("Can not apply gradients before zero_grad call.")
        if closure is not None:
            return self.optimizer_adam.step(closure)
        self.optimizer_adam.step()
        return None


# ============================================================
# Evaluation Helpers
# ============================================================

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
                              dataset_infos, num_node_types, reference_graphs=None,
                              train_graphs=None):
    fold_metrics = {}
    is_molecular = dataset_name in ('qm9', 'guacamol', 'zinc250k', 'moses')

    if is_molecular:
        atom_decoder = dataset_infos.get('atom_decoder')
        remove_h = dataset_infos.get('remove_h', True)
        train_graph_source = train_graphs if train_graphs is not None else graphs

        if atom_decoder is not None:
            print("Collecting training SMILES...")
            train_smiles = collect_train_smiles_molecular(train_graph_source, atom_decoder)

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
                fcd_score = compute_fcd(all_smiles, train_smiles)
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

        if dataset_name == 'sbm':
            sbm_proxy = metrics.get('sampling/frac_unique_non_iso')
            if sbm_proxy is None:
                sbm_proxy = metrics.get('sampling/frac_non_iso', metrics.get('frac_non_iso'))
            if sbm_proxy is not None:
                metrics.setdefault('sampling/frac_unique_non_iso', sbm_proxy)
                metrics.setdefault('frac_unique_non_iso', sbm_proxy)
                metrics.setdefault('sampling/frac_unic_non_iso_valid', sbm_proxy)
                metrics.setdefault('frac_unic_non_iso_valid', sbm_proxy)
                metrics.setdefault('valid', sbm_proxy)

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


# ============================================================
# Training Loss Wrapper
# ============================================================

class DeFoGWithLoss(WithLoss):

    r"""Wraps DeFoG model for GammaGL training with ``TrainOneStep``.

    Encapsulates the full training forward pass: noise application,
    extra feature computation, model forward, and loss computation.

    Parameters
    ----------
    backbone : DeFoGModel
        The graph transformer denoiser.
    loss_fn : TrainLossDiscrete
        The training loss function.
    noise_dist : NoiseDistribution
        Noise distribution handler.
    time_distorter : TimeDistorter
        Time distortion for sampling training time.
    extra_features : callable
        Structural extra feature computer.
    domain_features : callable
        Domain-specific extra feature computer.
    """
    def __init__(self, backbone, loss_fn, noise_dist, time_distorter,
                 extra_features, domain_features, conditional=False):
        super().__init__(backbone=backbone, loss_fn=loss_fn)
        self.noise_dist = noise_dist
        self.limit_dist = noise_dist.get_limit_dist()
        self.time_distorter = time_distorter
        self.extra_features = extra_features
        self.domain_features = domain_features
        self.conditional = conditional
        self._debug_forward_calls = 0

    def forward(self, data, label):
        """
        Parameters
        ----------
        data : dict
            Keys: ``'X'``, ``'E'``, ``'y'``, ``'node_mask'``.
        label : ignored

        Returns
        -------
        tensor
            Scalar loss value.
        """
        X = data['X']
        E = data['E']
        y = data['y']
        node_mask = data['node_mask']

        bs = X.shape[0]

        # Classifier-free guidance: 10% dropout of conditional labels
        if self.conditional and y.shape[-1] > 0:
            if np.random.rand() < 0.1:
                y = tlx.ones_like(y) * (-1.0)

        # Add virtual classes for absorbing transition
        X, E = self.noise_dist.add_virtual_classes(X, E)

        # 1. Apply noise
        noisy_data = apply_noise(X, E, y, node_mask,
                                  self.limit_dist, self.time_distorter)

        # 2. Compute extra features
        extra_data = compute_extra_data(noisy_data, self.extra_features,
                                         self.domain_features, self.noise_dist)

        # 3. Concatenate inputs
        X_in = tlx.concat([noisy_data['X_t'], extra_data.X], axis=-1)
        E_in = tlx.concat([noisy_data['E_t'], extra_data.E], axis=-1)

        # Ensure y tensors are 2D (bs, dy)
        y_t = noisy_data['y_t']
        ey = extra_data.y
        if len(y_t.shape) == 1:
            y_t = tlx.reshape(y_t, [bs, -1])
        if len(ey.shape) == 1:
            ey = tlx.reshape(ey, [bs, -1])
        if y_t.shape[-1] == 0:
            y_in = ey
        elif ey.shape[-1] == 0:
            y_in = y_t
        else:
            y_in = tlx.concat([y_t, ey], axis=-1)

        # 4. Forward through model
        pred_X, pred_E, pred_y = self.backbone_network(X_in, E_in, y_in, node_mask)

        self._debug_forward_calls += 1
        debug_this_call = self._debug_forward_calls <= 2

        if tlx.BACKEND == 'torch':
            import torch

            def _tensor_debug(name, tensor):
                if tensor is None or not hasattr(tensor, 'numel') or tensor.numel() == 0:
                    return True
                tensor_detached = tensor.detach()
                is_finite = bool(torch.isfinite(tensor_detached).all().item())
                if debug_this_call or not is_finite:
                    t_min = float(tensor_detached.min().item())
                    t_max = float(tensor_detached.max().item())
                    print(
                        f"[debug:loss] forward {self._debug_forward_calls} {name}: "
                        f"finite={is_finite} min={t_min:.6f} max={t_max:.6f}",
                        flush=True,
                    )
                return is_finite

            _tensor_debug('pred_X', pred_X)
            _tensor_debug('pred_E', pred_E)
            _tensor_debug('pred_y', pred_y)

        # 5. Compute loss against clean data (ignoring virtual classes)
        true_X, true_E = self.noise_dist.ignore_virtual_classes(X, E)

        loss = self._loss_fn(pred_X, pred_E, pred_y, true_X, true_E, y)

        if tlx.BACKEND == 'torch':
            import torch
            if debug_this_call or not torch.isfinite(loss):
                print(
                    f"[debug:loss] forward {self._debug_forward_calls} total_loss="
                    f"{float(loss.detach().item()):.6f} finite={bool(torch.isfinite(loss).item())}",
                    flush=True,
                )

        return loss


class MultiGPUTrainer:
    """Manual multi-GPU data parallelism using CUDA streams for parallel execution.

    Creates independent copies of the model+loss_wrapper on each GPU
    (via deepcopy), splits batches, runs forward+backward in parallel
    on separate CUDA streams, then averages gradients on the primary GPU.
    """
    def __init__(self, model, loss_wrapper, loss_fn, n_gpu, lr, weight_decay):
        import copy
        self.n_gpu = n_gpu
        self.primary = 0
        self.models = [model]
        self.loss_wrappers = [loss_wrapper]
        self.loss_fns = [loss_fn]

        # Ensure primary model is explicitly on cuda:0
        model.to('cuda:0')

        for i in range(1, n_gpu):
            m_copy = copy.deepcopy(model)
            m_copy.to(f'cuda:{i}')
            lw_copy = copy.deepcopy(loss_wrapper)
            lw_copy._backbone = m_copy
            self.models.append(m_copy)
            self.loss_wrappers.append(lw_copy)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay, amsgrad=True,
        )

        # Pre-allocate CUDA streams for each GPU
        self.streams = [torch.cuda.Stream(device=i) for i in range(n_gpu)]

    def train_step(self, data_dict):
        self.optimizer.zero_grad()

        # Split batch across GPUs
        X, E, y, mask = data_dict['X'], data_dict['E'], data_dict['y'], data_dict['node_mask']
        bs = X.shape[0]
        chunk_size = bs // self.n_gpu
        chunks = []
        start = 0
        for i in range(self.n_gpu):
            end = start + chunk_size + (1 if i < bs % self.n_gpu else 0)
            chunks.append((
                X[start:end].to(f'cuda:{i}', non_blocking=True),
                E[start:end].to(f'cuda:{i}', non_blocking=True),
                y[start:end].to(f'cuda:{i}', non_blocking=True),
                mask[start:end].to(f'cuda:{i}', non_blocking=True),
            ))
            start = end

        # Forward + backward on each GPU in parallel via CUDA streams
        per_gpu_losses = [None] * self.n_gpu
        per_gpu_grads = [None] * self.n_gpu
        errors = [None] * self.n_gpu

        def gpu_work(idx):
            try:
                stream = self.streams[idx]
                with torch.cuda.device(idx), torch.cuda.stream(stream):
                    Xi, Ei, yi, mi = chunks[idx]
                    d = {'X': Xi, 'E': Ei, 'y': yi, 'node_mask': mi}
                    loss = self.loss_wrappers[idx](d, None)
                    loss.backward()
                    per_gpu_losses[idx] = float(loss.item())
                    # Detach grads and move to primary GPU
                    grads = {}
                    for name, p in self.models[idx].named_parameters():
                        if p.grad is not None:
                            grads[name] = p.grad.to(f'cuda:{self.primary}', non_blocking=True).detach() / self.n_gpu
                    per_gpu_grads[idx] = grads
            except Exception as e:
                errors[idx] = e

        # Launch all GPU work in parallel via Python threads + CUDA streams
        threads = []
        for i in range(self.n_gpu):
            t = threading.Thread(target=gpu_work, args=(i,))
            t.start()
            threads.append(t)

        # Wait for all threads to finish
        for t in threads:
            t.join()

        # Check for errors
        for i, err in enumerate(errors):
            if err is not None:
                raise RuntimeError(f"GPU {i} failed: {err}") from err

        # Synchronize all streams to ensure computation is done
        for s in self.streams:
            s.synchronize()

        # Average loss
        avg_loss = sum(per_gpu_losses) / self.n_gpu

        # Accumulate all grads onto primary model
        torch.cuda.set_device(self.primary)
        for gpu_grads in per_gpu_grads:
            for name, p in self.models[self.primary].named_parameters():
                if name in gpu_grads:
                    if p.grad is None:
                        p.grad = gpu_grads[name]
                    else:
                        p.grad.add_(gpu_grads[name])

        # Optimizer step
        self.optimizer.step()

        # Sync weights to replicas
        with torch.no_grad():
            for i in range(1, self.n_gpu):
                for (_, p_src), (_, p_dst) in zip(
                    self.models[self.primary].named_parameters(),
                    self.models[i].named_parameters()
                ):
                    p_dst.data.copy_(p_src.data, non_blocking=True)
            # Synchronize weight copies
            for i in range(1, self.n_gpu):
                self.streams[i].synchronize()

        return avg_loss


# ============================================================
# Apply Noise (Forward Noising Process)
# ============================================================

def apply_noise(X, E, y, node_mask, limit_dist, time_distorter):
    r"""Apply noise to clean graph data for training.

    Parameters
    ----------
    X : tensor
        Clean node features ``(bs, n, dx)`` (one-hot).
    E : tensor
        Clean edge features ``(bs, n, n, de)`` (one-hot).
    y : tensor
        Global features ``(bs, dy)``.
    node_mask : tensor
        Boolean mask ``(bs, n)``.
    limit_dist : PlaceHolder
        Noise limit distribution.
    time_distorter : TimeDistorter
        Time distortion function.

    Returns
    -------
    dict
        Noisy data with keys ``'t'``, ``'X_t'``, ``'E_t'``, ``'y_t'``, ``'node_mask'``.
    """
    bs = X.shape[0]

    # Move limit_dist to the same device as input (needed for DataParallel)
    device = X.device if hasattr(X, 'device') else None
    if device is not None and hasattr(limit_dist, 'X') and hasattr(limit_dist.X, 'to'):
        if limit_dist.X.device != device:
            limit_dist = PlaceHolder(X=limit_dist.X.to(device), E=limit_dist.E.to(device), y=limit_dist.y)

    # Sample time
    t = time_distorter.train_ft(bs)  # (bs, 1)

    # Get clean integer labels
    X_1 = tlx.argmax(X, axis=-1)   # (bs, n)
    E_1 = tlx.argmax(E, axis=-1)   # (bs, n, n)

    # Compute transition probabilities
    prob_X, prob_E = p_xt_g_x1(X_1, E_1, t, limit_dist)
    # prob_X: (bs, n, dx), prob_E: (bs, n, n, de)

    # Sample noisy features
    sampled = sample_discrete_features(prob_X, prob_E, node_mask)
    X_t_int = sampled.X  # (bs, n)
    E_t_int = sampled.E  # (bs, n, n)

    dx = len(limit_dist.X)
    de = len(limit_dist.E)

    # One-hot encode
    X_t = backend_one_hot(X_t_int, dx)  # (bs, n, dx)
    E_t = backend_one_hot(E_t_int, de)  # (bs, n, n, de)

    # Mask
    x_mask = tlx.expand_dims(tlx.cast(node_mask, X_t.dtype), axis=-1)
    e_mask = tlx.expand_dims(x_mask, axis=2) * tlx.expand_dims(x_mask, axis=1)
    X_t = X_t * x_mask
    E_t = E_t * e_mask

    y_t = y if y is not None else tlx.zeros([bs, 0], dtype=tlx.float32)

    return {
        't': t,
        'X_t': X_t,
        'E_t': E_t,
        'y_t': y_t,
        'node_mask': node_mask,
    }


# ============================================================
# Compute Extra Data
# ============================================================

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
    PlaceHolder
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

    return PlaceHolder(X=extra_X, E=extra_E, y=extra_y)


# ============================================================
# Sampling
# ============================================================

def compute_step_probs(R_t_X, R_t_E, X_t, E_t, dt):
    r"""Convert rate matrices to one-step CTMC transition probabilities.

    Matches the original DeFoG logic: zero the current-state column first,
    then write back the stay probability so rows sum to 1.
    """
    step_X = R_t_X * dt
    step_E = R_t_E * dt

    cur_X = tlx.argmax(X_t, axis=-1)
    cur_E = tlx.argmax(E_t, axis=-1)

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
    if tlx.BACKEND == 'torch':
        import torch as _torch
        no_grad = _torch.no_grad
    else:
        from contextlib import nullcontext
        no_grad = nullcontext
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
                f"[debug:sampling] step {step_idx + 1}/{sample_steps} {tag} "
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
                f"[debug:sampling] step {step_idx + 1}/{sample_steps} {tag} "
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

        with no_grad():
            pred_X, pred_E, pred_y = model(X_in, E_in, y_in, node_mask)

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
                y_uncond = tlx.ones_like(y_t) * (-1.0)
                y_in_uncond = tlx.concat([y_uncond, extra_data.y], axis=-1)
                with no_grad():
                    pred_X_u, pred_E_u, _ = model(X_in, E_in, y_in_uncond, node_mask)
                pred_X_soft_u = tlx.softmax(pred_X_u, axis=-1)
                pred_E_soft_u = tlx.softmax(pred_E_u, axis=-1)

                eps_cfg = 1e-6
                w = guidance_weight
                pred_X_soft = tlx.softmax(
                    (1 - w) * tlx.log(pred_X_soft_u + eps_cfg)
                    + w * tlx.log(pred_X_soft + eps_cfg), axis=-1)
                pred_E_soft = tlx.softmax(
                    (1 - w) * tlx.log(pred_E_soft_u + eps_cfg)
                    + w * tlx.log(pred_E_soft + eps_cfg), axis=-1)

            sampled = sample_discrete_features(pred_X_soft, pred_E_soft, node_mask)
            X_t = backend_one_hot(sampled.X, dx)
            E_t = backend_one_hot(sampled.E, de)
        else:
            # Compute conditional rate matrix
            R_X, R_E = rate_matrix_designer.compute_graph_rate_matrix(
                t_dist, node_mask, (X_t, E_t), (pred_X_soft, pred_E_soft)
            )

            # Classifier-free guidance: blend rate matrices in log-space
            if conditional and cond_labels is not None:
                # Compute unconditional rate matrix
                y_uncond = tlx.ones_like(y_t) * (-1.0)
                y_in_uncond = tlx.concat([y_uncond, extra_data.y], axis=-1)
                with no_grad():
                    pred_X_u, pred_E_u, _ = model(X_in, E_in, y_in_uncond, node_mask)
                pred_X_soft_u = tlx.softmax(pred_X_u, axis=-1)
                pred_E_soft_u = tlx.softmax(pred_E_u, axis=-1)

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
            X_t = backend_one_hot(sampled.X, dx)
            E_t = backend_one_hot(sampled.E, de)

        # Mask
        x_mask = tlx.expand_dims(tlx.cast(node_mask, X_t.dtype), axis=-1)
        e_mask = tlx.expand_dims(x_mask, axis=2) * tlx.expand_dims(x_mask, axis=1)
        X_t = X_t * x_mask
        E_t = E_t * e_mask

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


# ============================================================
# Dataset Utilities
# ============================================================


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
    print(f"[debug:data] Enter compute_dataset_infos on {total_graphs} graphs", flush=True)
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
            print(f"[debug:data] compute_dataset_infos progress: {idx + 1}/{total_graphs}", flush=True)

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
        from gammagl.datasets.spectre_dataset import PlanarGraphDataset
        ds_cls = PlanarGraphDataset
        num_node_types, num_edge_types = 2, 2
        # Planar/Tree/SBM: x is ones(n,1), need to convert to one-hot for DeFoG
        convert_spectre = True
    elif name == 'tree':
        from gammagl.datasets.spectre_dataset import TreeGraphDataset
        ds_cls = TreeGraphDataset
        num_node_types, num_edge_types = 2, 2
        convert_spectre = True
    elif name == 'sbm':
        from gammagl.datasets.spectre_dataset import SBMGraphDataset
        ds_cls = SBMGraphDataset
        num_node_types, num_edge_types = 2, 2
        convert_spectre = True
    elif name == 'comm20':
        from gammagl.datasets.spectre_dataset import Comm20GraphDataset
        ds_cls = Comm20GraphDataset
        num_node_types, num_edge_types = 2, 2
        convert_spectre = True
    elif name == 'qm9':
        from gammagl.datasets.qm9_dataset import QM9Gen
        ds_cls = QM9Gen
        qm9_remove_h = True if remove_h is None else bool(remove_h)
        num_node_types, num_edge_types = (4, 5) if qm9_remove_h else (5, 5)
        convert_spectre = False
    elif name == 'guacamol':
        from gammagl.datasets.guacamol_dataset import GuacaMolDataset
        ds_cls = GuacaMolDataset
        num_node_types, num_edge_types = 12, 5
        convert_spectre = False
    elif name == 'zinc250k':
        from gammagl.datasets.zinc250k_dataset import ZINC250kGen
        ds_cls = ZINC250kGen
        num_node_types, num_edge_types = 9, 4
        convert_spectre = False
    elif name == 'moses':
        from gammagl.datasets.moses_dataset import MOSESDataset
        ds_cls = MOSESDataset
        num_node_types, num_edge_types = 8, 5
        convert_spectre = False
    elif name == 'tls':
        from gammagl.datasets.tls_dataset import TLSGraphDataset
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
        kwargs['use_defog_split'] = getattr(args, 'use_defog_split', False)
    # QM9 supports conditional generation
    if name == 'qm9' and conditional:
        kwargs['conditional'] = True
        kwargs['target'] = target
    print(f"[debug:data] Instantiating {name} train split...", flush=True)
    train_ds = ds_cls(split='train', **kwargs)
    print(f"[debug:data] Train split ready", flush=True)
    print(f"[debug:data] Instantiating {name} val split...", flush=True)
    val_ds = ds_cls(split='val', **kwargs)
    print(f"[debug:data] Val split ready", flush=True)
    print(f"[debug:data] Instantiating {name} test split...", flush=True)
    test_ds = ds_cls(split='test', **kwargs)
    print(f"[debug:data] Test split ready", flush=True)

    print(f"数据集 {name}: 训练集 {len(train_ds)}, 验证集 {len(val_ds)}, 测试集 {len(test_ds)}")

    # Convert spectre datasets: x=ones(n,1) → one-hot node type
    # edge_attr=[0,1] is already 2-class one-hot (no-edge, edge)
    if convert_spectre:
        train_graphs = []
        for i in range(len(train_ds)):
            g = train_ds[i]
            n = g.x.shape[0]
            # Node features: all same type → one-hot [1, 0] for type 0
            x_np = np.zeros((n, num_node_types), dtype=np.float32)
            x_np[:, 0] = 1.0
            g_new = Graph(
                x=tlx.convert_to_tensor(x_np),
                edge_index=g.edge_index,
                edge_attr=g.edge_attr,
                y=tlx.convert_to_tensor(np.zeros(1, dtype=np.float32)),
            )
            train_graphs.append(g_new)
    else:
        total_train = len(train_ds)
        print(f"[debug:data] Materializing train graphs for {name}: {total_train}", flush=True)
        train_graphs = []
        for i in range(total_train):
            g = train_ds[i]
            # Preserve y from dataset (conditional labels or empty)
            y_val = g.y if g.y is not None else np.zeros((1, 0), dtype=np.float32)
            if isinstance(y_val, np.ndarray):
                y_val = tlx.convert_to_tensor(y_val)
            g_new = Graph(
                x=g.x,
                edge_index=g.edge_index,
                edge_attr=g.edge_attr,
                y=y_val,
            )
            train_graphs.append(g_new)
            if (i + 1) <= 3 or (i + 1) % 50000 == 0 or (i + 1) == total_train:
                print(f"[debug:data] Materialize progress: {i + 1}/{total_train}", flush=True)

    print(f"[debug:data] Enter dataset info computation for {name}", flush=True)
    dataset_infos = compute_dataset_infos(train_graphs, num_node_types, num_edge_types)
    if name in ('qm9', 'guacamol', 'zinc250k', 'moses'):
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
    print(f"[debug:data] Dataset info computation done for {name}", flush=True)

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


# ============================================================
# Molecular Dataset Info Helper
# ============================================================

# ============================================================
# Main
# ============================================================

def main(args):
    # ------- Dataset -------
    if args.dataset == 'synthetic':
        print(f"Creating synthetic dataset with {args.num_graphs} graphs...")
        test_labels = None
        val_ds = None
        test_ds = None
        graphs = create_synthetic_dataset(
            num_graphs=args.num_graphs,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            num_node_types=args.num_node_types,
            num_edge_types=args.num_edge_types,
            p_edge=args.p_edge,
        )
        dataset_infos = compute_dataset_infos(graphs, args.num_node_types, args.num_edge_types)
    else:
        graphs, val_ds, test_ds, dataset_infos, nt, et, test_labels = load_real_dataset(
            args.dataset, root=args.data_root,
            conditional=getattr(args, 'conditional', False),
            target=getattr(args, 'target', 'mu'),
            remove_h=getattr(args, 'remove_h', None))
        args.num_node_types = nt
        args.num_edge_types = et

    val_labels = None
    if getattr(args, 'conditional', False) and val_ds is not None:
        val_labels_list = []
        for i in range(len(val_ds)):
            g = val_ds[i]
            y_val = g.y if g.y is not None else np.zeros((1, 0), dtype=np.float32)
            if isinstance(y_val, np.ndarray):
                y_np = y_val.flatten()
            else:
                y_np = tlx.convert_to_numpy(y_val).flatten()
            val_labels_list.append(y_np)
        if len(val_labels_list) > 0 and len(val_labels_list[0]) > 0:
            val_labels = tlx.convert_to_tensor(
                np.stack(val_labels_list, axis=0).astype(np.float32))

    print(f"Dataset: {len(graphs)} graphs, max_nodes={dataset_infos['max_n_nodes']}")
    print(f"  Node type distribution: {dataset_infos['node_types']}")
    print(f"  Edge type distribution: {dataset_infos['edge_types']}")

    # ------- Noise Distribution -------
    print("[debug:init] Enter NoiseDistribution...", flush=True)
    noise_dist = NoiseDistribution(args.transition, dataset_infos)
    limit_dist = noise_dist.get_limit_dist()
    print("[debug:init] NoiseDistribution ready", flush=True)

    # ------- Extra Features -------
    print("[debug:init] Enter ExtraFeatures...", flush=True)
    extra_features = ExtraFeatures(
        extra_features_type=args.extra_features,
        rrwp_steps=args.rrwp_steps,
        dataset_info=dataset_infos,
    )
    print("[debug:init] ExtraFeatures ready", flush=True)
    # ------- Domain-specific features -------
    is_molecular = args.dataset in ('qm9', 'guacamol', 'zinc250k', 'moses')
    if is_molecular:
        print("[debug:init] Enter ExtraMolecularFeatures...", flush=True)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        print("[debug:init] ExtraMolecularFeatures ready", flush=True)
    else:
        domain_features = DummyMolecularFeatures()

    # ------- Compute input dimensions by doing a dry run -------
    print("[debug:init] Enter dummy feature dry-run...", flush=True)
    # Create a dummy noisy data to infer feature sizes
    dummy_g = graphs[0]
    dummy_X = tlx.expand_dims(dummy_g.x, axis=0)  # (1, n, dx)
    dummy_n = dummy_X.shape[1]
    dummy_E_np = np.zeros((1, dummy_n, dummy_n, args.num_edge_types), dtype=np.float32)
    dummy_E_np[0, :, :, 0] = 1.0
    dummy_E = tlx.convert_to_tensor(dummy_E_np)
    dummy_mask = tlx.convert_to_tensor(np.ones((1, dummy_n), dtype=bool))
    # Determine y dimension: conditional datasets have non-empty y
    n_cond = 0
    if getattr(args, 'conditional', False) and test_labels is not None:
        n_cond = test_labels.shape[-1]
    dummy_y = tlx.zeros([1, n_cond], dtype=tlx.float32)
    dummy_t = tlx.convert_to_tensor(np.array([[0.5]], dtype=np.float32))

    # Add virtual classes
    print("[debug:init] Add virtual classes for dummy...", flush=True)
    dummy_X_v, dummy_E_v = noise_dist.add_virtual_classes(dummy_X, dummy_E)
    print("[debug:init] Dummy virtual classes ready", flush=True)

    dummy_noisy = {
        't': dummy_t,
        'X_t': dummy_X_v,
        'E_t': dummy_E_v,
        'y_t': dummy_y,
        'node_mask': dummy_mask,
    }
    print("[debug:init] Enter compute_extra_data(dummy)...", flush=True)
    extra_dummy = compute_extra_data(dummy_noisy, extra_features,
                                      domain_features, noise_dist)
    print("[debug:init] compute_extra_data(dummy) done", flush=True)

    input_X_dim = dummy_X_v.shape[-1] + extra_dummy.X.shape[-1]
    input_E_dim = dummy_E_v.shape[-1] + extra_dummy.E.shape[-1]
    input_y_dim = dummy_y.shape[-1] + extra_dummy.y.shape[-1]

    # Check for saved model_config.json (for loading checkpoints trained with different code)
    model_config_path = os.path.join(args.save_dir, 'model_config.json')
    if os.path.exists(model_config_path):
        import json as _json
        with open(model_config_path, 'r') as _f:
            _cfg = _json.load(_f)
        output_dims = _cfg.get('output_dims', noise_dist.get_noise_dims())
        print(f"[config] Loaded output_dims from {model_config_path}")
    else:
        # Use dataset_infos output_dims (matches original DeFoG behavior)
        output_dims = dataset_infos['output_dims']

    input_dims = {'X': input_X_dim, 'E': input_E_dim, 'y': input_y_dim}

    print(f"Input dims: {input_dims}")
    print(f"Output dims: {output_dims}")

    # ------- Model -------
    hidden_mlp_dims = {'X': args.hidden_mlp_X, 'E': args.hidden_mlp_E, 'y': args.hidden_mlp_y}
    hidden_dims = {
        'dx': args.dx, 'de': args.de, 'dy': args.dy,
        'n_head': args.n_head,
        'dim_ffX': args.dim_ffX, 'dim_ffE': args.dim_ffE, 'dim_ffy': args.dim_ffy,
    }

    model = DeFoGModel(
        n_layers=args.n_layers,
        input_dims=input_dims,
        hidden_mlp_dims=hidden_mlp_dims,
        hidden_dims=hidden_dims,
        output_dims=output_dims,
        name='DeFoG',
    )
    print(f"Model created with {args.n_layers} layers")

    # ------- Loss wrapper -------
    loss_fn = TrainLossDiscrete(
        lambda_train=[args.lambda_E, args.lambda_y],
        kld=getattr(args, 'kld', False),
    )

    conditional = getattr(args, 'conditional', False) and n_cond > 0
    loss_wrapper = DeFoGWithLoss(
        backbone=model,
        loss_fn=loss_fn,
        noise_dist=noise_dist,
        time_distorter=TimeDistorter(
            train_distortion=args.train_distortion,
            sample_distortion=args.sample_distortion,
        ),
        extra_features=extra_features,
        domain_features=domain_features,
        conditional=conditional,
    )

    # ------- Rate matrix designer -------
    rate_designer = RateMatrixDesigner(
        rdb=args.rdb,
        rdb_crit=args.rdb_crit,
        eta=args.eta,
        omega=args.omega,
        limit_dist=limit_dist,
    )

    # ------- Time distorter -------
    time_distorter = TimeDistorter(
        train_distortion=args.train_distortion,
        sample_distortion=args.sample_distortion,
    )

    # ------- Optimizer -------
    import torch as _torch
    optimizer = AdamW(
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )

    use_dp = getattr(args, 'n_gpu', 1) > 1
    multi_gpu = None

    if use_dp:
        multi_gpu = MultiGPUTrainer(
            model, loss_wrapper, loss_fn,
            n_gpu=args.n_gpu, lr=args.lr, weight_decay=args.weight_decay,
        )
        print(f"[MultiGPU] Using {args.n_gpu} GPUs")
        print(f"[MultiGPU] Effective batch_size={args.batch_size} "
              f"({args.batch_size // args.n_gpu} per GPU)")
    else:
        if args.grad_clip_norm is not None:
            optimizer.grad_clip = lambda weights: _torch.nn.utils.clip_grad_norm_(
                [w for w in weights], max_norm=args.grad_clip_norm)
        print("[debug] Creating TrainOneStep...")
        train_one_step = TrainOneStep(loss_wrapper, optimizer, model.trainable_weights)
        print("[debug] TrainOneStep created")

    # DataLoader with seeded shuffle
    print(f"[debug] Creating DataLoader with batch_size={args.batch_size}...")

    class SeededRandomSampler:
        """Reproducible random sampler: each epoch shuffles with seed + epoch."""
        def __init__(self, data_source, seed=42):
            self.data_source = data_source
            self.seed = seed
            self.epoch = 0
        def __iter__(self):
            rng = np.random.default_rng(self.seed + self.epoch)
            indices = np.arange(len(self.data_source))
            rng.shuffle(indices)
            self.epoch += 1
            for idx in indices:
                yield int(idx)
        def __len__(self):
            return len(self.data_source)

    from tensorlayerx.dataflow import BatchSampler
    from gammagl.loader.dataloader import Collater
    sampler = SeededRandomSampler(graphs, seed=args.seed)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=False)
    loader = DataLoader(graphs, batch_sampler=batch_sampler, collate_fn=Collater(follow_batch=None, exclude_keys=None))
    print("[debug] DataLoader created (seeded shuffle)")

    # EMA (Exponential Moving Average)
    ema = None
    if args.ema_decay > 0:
        ema = EMA(model, decay=args.ema_decay)
        print(f"EMA enabled with decay={args.ema_decay}")

    if args.sample:
        print("\nSkipping training because --sample was specified.")
    else:
        # Resume from checkpoint if specified
        start_epoch = getattr(args, 'start_epoch', 0) or 0
        if args.resume_from:
            ckpt_path = os.path.join(args.resume_from, 'last_model.npz')
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(args.resume_from, 'best_model.npz')
            if os.path.exists(ckpt_path):
                model.load_weights(ckpt_path, format='npz_dict')
                print(f"Resumed model weights from {ckpt_path}, starting at epoch {start_epoch}")
            else:
                print(f"WARNING: --resume_from={args.resume_from} but no checkpoint found, training from scratch")

        print(f"\nStarting training for {args.n_epochs} epochs (from epoch {start_epoch})...")
        best_score = float('-inf')
        best_epoch = None
        best_metrics = None
        val_counter = 0

        for epoch in range(start_epoch, args.n_epochs):
            model.set_train()
            if not use_dp:
                loss_fn.reset()
            total_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(loader):
                if epoch == 0 and batch_idx == 0:
                    print("[debug:train] Fetched first batch from DataLoader", flush=True)
                    print("[debug:train] Enter to_dense...", flush=True)
                # Convert sparse batch to dense
                dense, node_mask = to_dense(batch.x, batch.edge_index,
                                             batch.edge_attr, batch.batch)
                bs = dense.X.shape[0]
                if epoch == 0 and batch_idx == 0:
                    print(f"[debug:train] to_dense done: bs={bs}, n={dense.X.shape[1]}", flush=True)

                # Extract y (conditional labels or empty)
                if conditional and hasattr(batch, 'y') and batch.y is not None:
                    y_np = tlx.convert_to_numpy(batch.y)
                    if y_np.ndim == 1:
                        y_np = y_np.reshape(bs, -1)
                    elif y_np.ndim > 2:
                        y_np = y_np.reshape(bs, -1)
                    # Filter out dummy y values (shape[1]==0)
                    if y_np.shape[-1] == 0:
                        y = tlx.zeros([bs, 0], dtype=tlx.float32)
                    else:
                        y = tlx.convert_to_tensor(y_np.astype(np.float32))
                else:
                    y = tlx.zeros([bs, 0], dtype=tlx.float32)

                data_dict = {
                    'X': dense.X,
                    'E': dense.E,
                    'y': y,
                    'node_mask': node_mask,
                }

                if y.shape[-1] > 0 and y.dtype != tlx.float32:
                    y = tlx.cast(y, tlx.float32)
                    data_dict['y'] = y

                if epoch == 0 and batch_idx == 0:
                    print("[debug:train] Enter first train step...", flush=True)

                if use_dp:
                    loss_val = multi_gpu.train_step(data_dict)
                else:
                    loss = train_one_step(data_dict, None)
                    loss_val = float(loss) if isinstance(loss, (int, float)) else \
                        float(loss.item() if hasattr(loss, 'item') else np.asarray(loss).item())

                if epoch == 0 and batch_idx == 0:
                    print("[debug:train] First train step done", flush=True)
                total_loss += loss_val
                n_batches += 1

                if epoch == 0 and ((batch_idx + 1) <= 5 or (batch_idx + 1) % 20 == 0):
                    print(
                        f"[debug:train] Epoch 1 progress: batch {batch_idx + 1}, "
                        f"loss={loss_val:.4f}",
                        flush=True,
                    )

                # Update EMA after each training step
                if ema is not None:
                    ema.update(model)

            avg_loss = total_loss / max(n_batches, 1)

            if use_dp:
                print(f"  Epoch {epoch + 1}/{args.n_epochs}: loss={avg_loss:.4f} "
                      f"(MultiGPU x{args.n_gpu})")
            else:
                epoch_metrics = loss_fn.log_epoch_metrics()
                print(f"  Epoch {epoch + 1}/{args.n_epochs}: loss={avg_loss:.4f} "
                      f"X_CE={epoch_metrics['x_CE']:.4f} E_CE={epoch_metrics['E_CE']:.4f}")

            should_validate = (
                args.check_val_every_n_epochs > 0 and
                (epoch + 1) % args.check_val_every_n_epochs == 0
            )

            if should_validate:
                val_counter += 1
                save_model_snapshot(model, ema, args.save_dir, 'last', output_dims)

                if args.sample_every_val > 0 and val_counter % args.sample_every_val == 0:
                    print(f"\nValidation sampling at epoch {epoch + 1}...")
                    val_batch_size = max(1, int(args.val_num_samples))
                    cond_labels = None
                    if conditional and val_labels is not None:
                        perm = np.random.permutation(tlx.convert_to_numpy(val_labels).shape[0])
                        idx = perm[:val_batch_size]
                        cond_labels = val_labels[idx]
                        print(f"  Using classifier-free guidance (weight={args.guidance_weight})")

                    if ema is not None:
                        ema.swap_in(model)
                    try:
                        generated_val = sample_batch(
                            model=model,
                            noise_dist=noise_dist,
                            rate_matrix_designer=rate_designer,
                            time_distorter=time_distorter,
                            extra_features=extra_features,
                            domain_features=domain_features,
                            node_dist=dataset_infos['node_dist'],
                            sample_steps=args.sample_steps,
                            batch_size=val_batch_size,
                            conditional=conditional,
                            cond_labels=cond_labels,
                            guidance_weight=args.guidance_weight,
                        )
                        val_metrics = evaluate_generated_graphs(
                            generated_val,
                            args.dataset,
                            graphs,
                            val_ds,
                            dataset_infos,
                            args.num_node_types,
                            reference_graphs=[val_ds[i] for i in range(min(len(val_ds), 200))] if val_ds is not None else None,
                            train_graphs=graphs,
                        )
                    finally:
                        if ema is not None:
                            ema.swap_out(model)

                    score = compute_selection_score(args.dataset, val_metrics)
                    print(f"  Validation selection score: {score:.6f}")
                    if score >= best_score:
                        best_score = score
                        best_epoch = epoch + 1
                        best_metrics = dict(val_metrics)
                        save_model_snapshot(model, ema, args.save_dir, 'best', output_dims)
                        print(f"  Updated best checkpoint at epoch {best_epoch}")

        save_model_snapshot(model, ema, args.save_dir, 'last', output_dims)
        if best_epoch is None:
            save_model_snapshot(model, ema, args.save_dir, 'best', output_dims)
            print("\nTraining complete. No validation-selected best checkpoint was produced; using last snapshot as best.")
        else:
            print(f"\nTraining complete. Best validation score {best_score:.6f} at epoch {best_epoch}.")
            if best_metrics is not None:
                print(f"  Best metrics keys: {sorted(best_metrics.keys())}")

    # ------- Sampling & Evaluation -------
    if args.sample:
        _, sampling_ema = load_model_snapshot_for_sampling(
            model, args.save_dir, ema_decay=args.ema_decay)

        num_folds = max(1, args.num_sample_fold)
        all_fold_metrics = []

        for fold in range(num_folds):
            print(f"\n--- Sampling fold {fold + 1}/{num_folds} ---")
            print(f"Generating {args.num_samples} graphs with {args.sample_steps} steps...")

            cond_labels = None
            if conditional and test_labels is not None:
                perm = np.random.permutation(tlx.convert_to_numpy(test_labels).shape[0])
                idx = perm[:args.num_samples]
                cond_labels = test_labels[idx]
                print(f"  Using classifier-free guidance (weight={args.guidance_weight})")

            generated = sample_batch(
                model=model,
                noise_dist=noise_dist,
                rate_matrix_designer=rate_designer,
                time_distorter=time_distorter,
                extra_features=extra_features,
                domain_features=domain_features,
                node_dist=dataset_infos['node_dist'],
                sample_steps=args.sample_steps,
                batch_size=args.num_samples,
                conditional=conditional,
                cond_labels=cond_labels,
                guidance_weight=args.guidance_weight,
            )

            print(f"Generated {len(generated)} graphs:")
            for i, (x, e) in enumerate(generated[:5]):
                n_edges = int(np.sum(e > 0)) // 2
                print(f"  Graph {i}: {len(x)} nodes, {n_edges} edges")

            save_name = f'generated_graphs_fold{fold}.npy' if num_folds > 1 else 'generated_graphs.npy'
            save_path = os.path.join(args.save_dir, save_name)
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(generated, f)
            print(f"Saved to {save_path}")

            if args.evaluate:
                print("\n" + "=" * 60)
                print(f"EVALUATION (fold {fold + 1}/{num_folds})")
                print("=" * 60)
                fold_metrics = evaluate_generated_graphs(
                    generated,
                    args.dataset,
                    graphs,
                    test_ds,
                    dataset_infos,
                    args.num_node_types,
                )
                all_fold_metrics.append(fold_metrics)

        if num_folds > 1 and len(all_fold_metrics) > 1:
            print("\n" + "=" * 60)
            print(f"MULTI-FOLD SUMMARY ({num_folds} folds)")
            print("=" * 60)
            all_keys = sorted(set().union(*all_fold_metrics))
            for key in all_keys:
                vals = [m[key] for m in all_fold_metrics if key in m]
                if vals:
                    mean_val = np.mean(vals)
                    std_val = np.std(vals)
                    print(f"  {key}: {mean_val:.6f} +/- {std_val:.6f}")

        if sampling_ema is not None:
            sampling_ema.swap_out(model)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeFoG: Discrete Flow Matching for Graph Generation')

    # Dataset
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['synthetic', 'planar', 'tree', 'sbm', 'comm20', 'qm9',
                                 'guacamol', 'zinc250k', 'moses', 'tls'],
                        help='Dataset to use')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory for real datasets')
    qm9_h_group = parser.add_mutually_exclusive_group()
    qm9_h_group.add_argument('--remove_h', dest='remove_h', action='store_true',
                             help='Use QM9 without hydrogens')
    qm9_h_group.add_argument('--with_h', dest='remove_h', action='store_false',
                             help='Use QM9 with hydrogens')
    parser.set_defaults(remove_h=None)
    parser.add_argument('--use_defog_split', action='store_true',
                        help='Use DeFoG original CSV split for QM9 instead of random split')
    parser.add_argument('--num_graphs', type=int, default=200)
    parser.add_argument('--min_nodes', type=int, default=10)
    parser.add_argument('--max_nodes', type=int, default=20)
    parser.add_argument('--num_node_types', type=int, default=2)
    parser.add_argument('--num_edge_types', type=int, default=2)
    parser.add_argument('--p_edge', type=float, default=0.3)

    # Conditional generation (classifier-free guidance)
    parser.add_argument('--conditional', action='store_true',
                        help='Enable classifier-free guidance conditional generation')
    parser.add_argument('--target', type=str, default='mu',
                        choices=['mu', 'homo', 'both'],
                        help='Target property for conditional generation (QM9 only)')
    parser.add_argument('--guidance_weight', type=float, default=2.0,
                        help='Classifier-free guidance weight')

    # Model
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--hidden_mlp_X', type=int, default=256)
    parser.add_argument('--hidden_mlp_E', type=int, default=128)
    parser.add_argument('--hidden_mlp_y', type=int, default=128)
    parser.add_argument('--dx', type=int, default=256)
    parser.add_argument('--de', type=int, default=64)
    parser.add_argument('--dy', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--dim_ffX', type=int, default=256)
    parser.add_argument('--dim_ffE', type=int, default=128)
    parser.add_argument('--dim_ffy', type=int, default=128)

    # Training
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-12)
    parser.add_argument('--grad_clip_norm', type=float, default=None,
                        help='Max norm for gradient clipping (default: disabled, matching original DeFoG)')
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='EMA decay (0 = disabled, typical: 0.999)')
    parser.add_argument('--kld', action='store_true',
                        help='Use KL-divergence loss instead of cross-entropy')
    parser.add_argument('--lambda_E', type=float, default=5.0)
    parser.add_argument('--lambda_y', type=float, default=0.0)
    parser.add_argument('--transition', type=str, default='marginal')
    parser.add_argument('--extra_features', type=str, default='rrwp')
    parser.add_argument('--rrwp_steps', type=int, default=12)
    parser.add_argument('--train_distortion', type=str, default='identity')

    # Sampling
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate generated graphs (molecular or synthetic metrics)')
    parser.add_argument('--sample_steps', type=int, default=100)
    parser.add_argument('--sample_distortion', type=str, default='identity')
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--omega', type=float, default=0.0)
    parser.add_argument('--rdb', type=str, default='general')
    parser.add_argument('--rdb_crit', type=str, default='max_marginal')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--num_sample_fold', type=int, default=1,
                        help='Number of sampling folds for evaluation (reports mean±std)')
    parser.add_argument('--sample_every_val', type=int, default=0,
                        help='Run validation sampling every N validation events (0 = disabled)')
    parser.add_argument('--check_val_every_n_epochs', type=int, default=0,
                        help='Run validation cadence every N epochs (0 = disabled)')
    parser.add_argument('--val_num_samples', type=int, default=40,
                        help='Number of samples used during validation selection')

    # Resume
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Checkpoint directory to resume from (loads last_model.npz)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Epoch to start from (0-indexed, used with --resume_from)')

    # System
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='Number of GPUs for DataParallel (1 = single GPU, no DP)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    args = apply_dataset_preset(args, parser)

    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    if args.gpu >= 0:
        tlx.set_device('GPU', args.gpu)
    else:
        tlx.set_device('CPU')

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
