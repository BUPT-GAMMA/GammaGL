import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
import random
import warnings
import numpy as np
import tensorlayerx as tlx
assert tlx.BACKEND == 'torch', "DeFoG currently only supports PyTorch backend due to framework limitations."
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

from gammagl.loader import DataLoader

from dataset_utils import create_synthetic_dataset, load_real_dataset, compute_dataset_infos
from defog_utils import PlaceHolder, to_dense, EMA

from flow_matching import NoiseDistribution, apply_noise, RateMatrixDesigner, TimeDistorter

from extra_features import ExtraFeatures, compute_extra_data, DummyExtraFeatures, ExtraMolecularFeatures

from train_metrics import TrainLossDiscrete
from sampler import sample_batch
from evaluator import evaluate_generated_graphs, compute_selection_score



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
        'de': 64,
        'dy': 64,
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
        'de': 64,
        'dy': 128,
        'dim_ffX': 256,
        'dim_ffE': 128,
        'dim_ffy': 256,
        'n_epochs': 300,
        'batch_size': 256,
        'train_distortion': 'polydec',
        'sample_distortion': 'polydec',
        'sample_steps': 1000,
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
        'de': 128,
        'dy': 128,
        'dim_ffX': 256,
        'dim_ffE': 128,
        'dim_ffy': 256,
        'n_epochs': 1000,
        'batch_size': 64,
        'train_distortion': 'polydec',
        'sample_distortion': 'polydec',
        'sample_steps': 1000,
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
        'de': 128,
        'dy': 128,
        'dim_ffX': 256,
        'dim_ffE': 128,
        'dim_ffy': 256,
        'n_epochs': 300,
        'batch_size': 256,
        'train_distortion': 'polydec',
        'sample_distortion': 'polydec',
        'sample_steps': 1000,
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

        # 5. Compute loss against clean data (ignoring virtual classes)
        true_X, true_E = self.noise_dist.ignore_virtual_classes(X, E)

        loss = self._loss_fn(pred_X, pred_E, pred_y, true_X, true_E, y)

        import torch
        if not torch.isfinite(loss):
            print(
                f"Warning: Non-finite loss ({loss}) encountered at step {self.global_step}. Skipping step."
            )
            # In pure PyTorch we might skip optimizer.step(), but here we just return the loss

        return loss

# ============================================================
# Dataset Utilities
# ============================================================




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
    noise_dist = NoiseDistribution(args.transition, dataset_infos)
    limit_dist = noise_dist.get_limit_dist()

    # ------- Extra Features -------
    extra_features = ExtraFeatures(
        extra_features_type=args.extra_features,
        rrwp_steps=args.rrwp_steps,
        dataset_info=dataset_infos,
    )
    # ------- Domain-specific features -------
    is_molecular = args.dataset in ('qm9', 'guacamol', 'zinc250k', 'moses')
    if is_molecular:
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        domain_features = DummyExtraFeatures()

    # ------- Compute input dimensions by doing a dry run -------
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
    dummy_X_v, dummy_E_v = noise_dist.add_virtual_classes(dummy_X, dummy_E)

    dummy_noisy = {
        't': dummy_t,
        'X_t': dummy_X_v,
        'E_t': dummy_E_v,
        'y_t': dummy_y,
        'node_mask': dummy_mask,
    }
    extra_dummy = compute_extra_data(dummy_noisy, extra_features,
                                      domain_features, noise_dist)

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

    # ------- Time distorter -------
    time_distorter = TimeDistorter(
        train_distortion=args.train_distortion,
        sample_distortion=args.sample_distortion,
    )

    conditional = getattr(args, 'conditional', False) and n_cond > 0
    loss_wrapper = DeFoGWithLoss(
        backbone=model,
        loss_fn=loss_fn,
        noise_dist=noise_dist,
        time_distorter=time_distorter,
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

    # ------- Optimizer -------
    import torch as _torch

    # AdamW wrapper: provides the tlx.optimizers interface around torch.optim.AdamW
    # with NaN/Inf gradient sanitization.
    class _AdamWWrapper(tlx.optimizers.Adam):
        def __init__(self, lr, weight_decay, amsgrad, grad_clip=None):
            self.amsgrad = amsgrad
            super().__init__(lr=lr, weight_decay=weight_decay, grad_clip=grad_clip)
        def gradient(self, loss, weights=None, return_grad=True):
            if weights is None:
                raise AttributeError("Parameter train_weights must be entered.")
            if not self.init_optim:
                self.optimizer_adam = _torch.optim.AdamW(
                    params=weights, lr=self.lr,
                    betas=(self.beta_1, self.beta_2), eps=self.eps,
                    weight_decay=self.weight_decay, amsgrad=self.amsgrad)
                self.init_optim = True
            self.optimizer_adam.zero_grad()
            if not _torch.isfinite(loss):
                print("[warn:optim] Non-finite loss; skipping step", flush=True)
                return [_torch.zeros_like(w) for w in weights] if return_grad else None
            loss.backward()
            for w in weights:
                if w.grad is not None and not _torch.isfinite(w.grad).all():
                    w.grad.data = _torch.nan_to_num(w.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
            if self.grad_clip is not None:
                gn = self.grad_clip(weights)
                if isinstance(gn, _torch.Tensor) and not _torch.isfinite(gn):
                    for w in weights:
                        if w.grad is not None:
                            w.grad.zero_()
            return [w.grad for w in weights] if return_grad else None
        def apply_gradients(self, grads_and_vars=None, closure=None):
            if not self.init_optim:
                raise AttributeError("Call gradient() first.")
            return self.optimizer_adam.step(closure) if closure else self.optimizer_adam.step()

    optimizer = _AdamWWrapper(
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )

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

    try:
        import torch
        from torch.utils.data import DataLoader as TorchDataLoader
        loader = TorchDataLoader(
            graphs,
            batch_sampler=batch_sampler,
            collate_fn=Collater(follow_batch=None, exclude_keys=None),
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            multiprocessing_context='spawn'
        )
        print("[debug] DataLoader created (seeded shuffle) with num_workers=8 (PyTorch)")
    except Exception as e:
        print(f"[warn] PyTorch DataLoader failed: {e}. Falling back to default.")
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
            ema_path = os.path.join(args.resume_from, 'last_ema.pkl')
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(args.resume_from, 'best_model.npz')
                ema_path = os.path.join(args.resume_from, 'best_ema.pkl')
            if os.path.exists(ckpt_path):
                model.load_weights(ckpt_path, format='npz_dict')
                print(f"Resumed model weights from {ckpt_path}, starting at epoch {start_epoch}")
                if ema is not None and os.path.exists(ema_path):
                    with open(ema_path, 'rb') as f:
                        ema.load_state_dict(f.read())
                    print(f"Resumed EMA weights from {ema_path}")
            else:
                print(f"WARNING: --resume_from={args.resume_from} but no checkpoint found, training from scratch")

        print(f"\nStarting training for {args.n_epochs} epochs (from epoch {start_epoch})...")
        saved_checkpoints = []
        max_saved_checkpoints = 5
        val_counter = 0

        for epoch in range(start_epoch, args.n_epochs):
            model.set_train()
            loss_fn.reset()
            total_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(loader):
                batch.tensor()
                # Convert sparse batch to dense
                dense, node_mask = to_dense(batch.x, batch.edge_index,
                                             batch.edge_attr, batch.batch)
                bs = dense.X.shape[0]

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

                loss = train_one_step(data_dict, None)
                loss_val = float(loss) if isinstance(loss, (int, float)) else \
                    float(loss.item() if hasattr(loss, 'item') else np.asarray(loss).item())
                total_loss += loss_val
                n_batches += 1

                if batch_idx % 20 == 0:
                    print(f"  Epoch {epoch + 1}, Batch {batch_idx}, loss={loss_val:.4f}")

                # Update EMA after each training step
                if ema is not None:
                    ema.update(model)

            avg_loss = total_loss / max(n_batches, 1)

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
                        sample_bs = getattr(args, 'sample_batch_size', 0) or val_batch_size
                        if sample_bs >= val_batch_size:
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
                        else:
                            all_generated_batches = []
                            num_batches = (val_batch_size + sample_bs - 1) // sample_bs
                            for b_idx in range(num_batches):
                                current_bs = min(sample_bs, val_batch_size - len(all_generated_batches))
                                print(f"  Validation batch {b_idx + 1}/{num_batches} (size={current_bs})...")
                                batch_cond = None
                                if cond_labels is not None:
                                    start_idx = b_idx * sample_bs
                                    batch_cond = cond_labels[start_idx:start_idx + current_bs]
                                batch_generated = sample_batch(
                                    model=model,
                                    noise_dist=noise_dist,
                                    rate_matrix_designer=rate_designer,
                                    time_distorter=time_distorter,
                                    extra_features=extra_features,
                                    domain_features=domain_features,
                                    node_dist=dataset_infos['node_dist'],
                                    sample_steps=args.sample_steps,
                                    batch_size=current_bs,
                                    conditional=conditional,
                                    cond_labels=batch_cond,
                                    guidance_weight=args.guidance_weight,
                                )
                                all_generated_batches.extend(batch_generated)
                            generated_val = all_generated_batches

                        val_metrics = evaluate_generated_graphs(
                            generated_val,
                            args.dataset,
                            graphs,
                            val_ds,
                            dataset_infos,
                            reference_graphs=[val_ds[i] for i in range(min(len(val_ds), 200))] if val_ds is not None else None,
                            train_graphs=graphs,
                            cache_dir=args.save_dir,
                            cond_labels=cond_labels,
                        )
                    finally:
                        if ema is not None:
                            ema.swap_out(model)

                    # Save rolling N checkpoints instead of relying on selection_score
                    ckpt_prefix = f'epoch_{epoch + 1}'
                    save_model_snapshot(model, ema, args.save_dir, ckpt_prefix, output_dims)
                    saved_checkpoints.append(ckpt_prefix)
                    print(f"  Saved checkpoint at epoch {epoch + 1}")

                    if len(saved_checkpoints) > max_saved_checkpoints:
                        old_prefix = saved_checkpoints.pop(0)
                        old_model_path = os.path.join(args.save_dir, f'{old_prefix}_model.npz')
                        old_ema_path = os.path.join(args.save_dir, f'{old_prefix}_ema.pkl')
                        if os.path.exists(old_model_path):
                            os.remove(old_model_path)
                        if os.path.exists(old_ema_path):
                            os.remove(old_ema_path)
                        print(f"  Removed old checkpoint {old_prefix}")

        save_model_snapshot(model, ema, args.save_dir, 'last', output_dims)
        print("\nTraining complete. Last snapshot saved.")

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

            # Support memory-constrained sampling via sample_batch_size
            sample_bs = getattr(args, 'sample_batch_size', 0) or args.num_samples
            if sample_bs >= args.num_samples:
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
            else:
                all_generated_batches = []
                num_batches = (args.num_samples + sample_bs - 1) // sample_bs
                for b_idx in range(num_batches):
                    current_bs = min(sample_bs, args.num_samples - len(all_generated_batches))
                    print(f"  Sampling batch {b_idx + 1}/{num_batches} (size={current_bs})...")
                    batch_cond = None
                    if cond_labels is not None:
                        start = b_idx * sample_bs
                        end = start + current_bs
                        batch_cond = cond_labels[start:end]
                    batch_generated = sample_batch(
                        model=model,
                        noise_dist=noise_dist,
                        rate_matrix_designer=rate_designer,
                        time_distorter=time_distorter,
                        extra_features=extra_features,
                        domain_features=domain_features,
                        node_dist=dataset_infos['node_dist'],
                        sample_steps=args.sample_steps,
                        batch_size=current_bs,
                        conditional=conditional,
                        cond_labels=batch_cond,
                        guidance_weight=args.guidance_weight,
                    )
                    all_generated_batches.extend(batch_generated)
                generated = all_generated_batches

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
                    cache_dir=args.save_dir,
                    cond_labels=cond_labels,
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
                        choices=['mu', 'homo', 'both', 'k2'],
                        help='Target property for conditional generation (QM9: mu/homo/both; TLS: k2)')
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
    parser.add_argument('--sample_batch_size', type=int, default=0,
                        help='Batch size for sampling (0 = use num_samples, for memory-constrained evaluation)')
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
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    args = apply_dataset_preset(args, parser)

    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    import torch
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
