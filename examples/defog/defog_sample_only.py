"""Standalone sampling-only script for DeFoG.
Loads a trained checkpoint and runs sampling + evaluation without training.

Usage:
    python defog_sample_only.py \
        --dataset planar \
        --gpu 3 \
        --save_dir ./checkpoints_planar \
        --sample_steps 1000 \
        --sample_distortion polydec \
        --omega 0.05 --eta 50 \
        --num_samples 512 \
        --num_sample_fold 3 \
        --evaluate
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
import numpy as np
import tensorlayerx as tlx

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, CURRENT_DIR)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

from gammagl.models.defog import DeFoGModel
from noise_distribution import NoiseDistribution
from rate_matrix import RateMatrixDesigner
from time_distorter import TimeDistorter
from extra_features import ExtraFeatures
from extra_features_molecular import ExtraMolecularFeatures, DummyMolecularFeatures

# Import shared helpers from defog_trainer
from defog_trainer import (
    load_real_dataset,
    compute_extra_data,
    sample_batch,
    apply_dataset_preset,
    load_model_snapshot_for_sampling,
    evaluate_generated_graphs,
)


def main(args):
    # ------- Dataset -------
    graphs, val_ds, test_ds, dataset_infos, nt, et, test_labels = load_real_dataset(
        args.dataset, root=args.data_root,
        conditional=getattr(args, 'conditional', False),
        target=getattr(args, 'target', 'mu'),
        remove_h=getattr(args, 'remove_h', None))
    args.num_node_types = nt
    args.num_edge_types = et

    print(f"Dataset: {len(graphs)} graphs, max_nodes={dataset_infos['max_n_nodes']}")

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
        domain_features = DummyMolecularFeatures()

    # ------- Compute input dimensions (same dry-run as training) -------
    dummy_g = graphs[0]
    dummy_X = tlx.expand_dims(dummy_g.x, axis=0)
    dummy_n = dummy_X.shape[1]
    dummy_E_np = np.zeros((1, dummy_n, dummy_n, args.num_edge_types), dtype=np.float32)
    dummy_E_np[0, :, :, 0] = 1.0
    dummy_E = tlx.convert_to_tensor(dummy_E_np)
    dummy_mask = tlx.convert_to_tensor(np.ones((1, dummy_n), dtype=bool))

    n_cond = 0
    if getattr(args, 'conditional', False) and test_labels is not None:
        n_cond = test_labels.shape[-1]
    dummy_y = tlx.zeros([1, n_cond], dtype=tlx.float32)
    dummy_t = tlx.convert_to_tensor(np.array([[0.5]], dtype=np.float32))

    dummy_X_v, dummy_E_v = noise_dist.add_virtual_classes(dummy_X, dummy_E)
    dummy_noisy = {
        't': dummy_t,
        'X_t': dummy_X_v,
        'E_t': dummy_E_v,
        'y_t': dummy_y,
        'node_mask': dummy_mask,
    }
    extra_dummy = compute_extra_data(dummy_noisy, extra_features, domain_features, noise_dist)

    input_X_dim = dummy_X_v.shape[-1] + extra_dummy.X.shape[-1]
    input_E_dim = dummy_E_v.shape[-1] + extra_dummy.E.shape[-1]
    input_y_dim = dummy_y.shape[-1] + extra_dummy.y.shape[-1]
    output_dims = dataset_infos.get('output_dims', noise_dist.get_noise_dims())

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

    # ------- Time Distorter -------
    time_distorter = TimeDistorter(
        train_distortion=args.train_distortion,
        sample_distortion=args.sample_distortion,
    )

    # ------- Load checkpoint -------
    _, sampling_ema = load_model_snapshot_for_sampling(
        model, args.save_dir, ema_decay=args.ema_decay)

    # ------- Sampling -------
    conditional = getattr(args, 'conditional', False) and n_cond > 0
    rate_designer = RateMatrixDesigner(
        rdb=args.rdb,
        rdb_crit=args.rdb_crit,
        eta=args.eta,
        omega=args.omega,
        limit_dist=limit_dist,
    )

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

        # Save generated graphs
        import pickle
        save_name = f'generated_graphs_fold{fold}.npy' if num_folds > 1 else 'generated_graphs.npy'
        save_path = os.path.join(args.save_dir, save_name)
        with open(save_path, 'wb') as f:
            pickle.dump(generated, f)
        print(f"Saved to {save_path}")

        # ------- Evaluation -------
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

    # Multi-fold summary
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

    # Restore original weights
    if sampling_ema is not None:
        sampling_ema.swap_out(model)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeFoG Sampling Only')

    # Dataset
    parser.add_argument('--dataset', type=str, default='planar',
                        choices=['synthetic', 'planar', 'tree', 'sbm', 'comm20', 'qm9',
                                 'guacamol', 'zinc250k', 'moses', 'tls'])
    parser.add_argument('--data_root', type=str, default=None)
    qm9_h_group = parser.add_mutually_exclusive_group()
    qm9_h_group.add_argument('--remove_h', dest='remove_h', action='store_true',
                             help='Use QM9 without hydrogens')
    qm9_h_group.add_argument('--with_h', dest='remove_h', action='store_false',
                             help='Use QM9 with hydrogens')
    parser.set_defaults(remove_h=None)

    # Conditional generation
    parser.add_argument('--conditional', action='store_true')
    parser.add_argument('--target', type=str, default='mu',
                        choices=['mu', 'homo', 'both'])
    parser.add_argument('--guidance_weight', type=float, default=2.0)

    # Model (must match training config)
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

    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--transition', type=str, default='marginal')
    parser.add_argument('--extra_features', type=str, default='rrwp')
    parser.add_argument('--rrwp_steps', type=int, default=12)
    parser.add_argument('--train_distortion', type=str, default='identity')

    # Sampling
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--sample_steps', type=int, default=1000)
    parser.add_argument('--sample_distortion', type=str, default='identity')
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--omega', type=float, default=0.0)
    parser.add_argument('--rdb', type=str, default='general')
    parser.add_argument('--rdb_crit', type=str, default='max_marginal')
    parser.add_argument('--num_samples', type=int, default=512)
    parser.add_argument('--num_sample_fold', type=int, default=1)

    # System
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    args = apply_dataset_preset(args, parser)

    np.random.seed(args.seed)

    if args.gpu >= 0:
        tlx.set_device('GPU', args.gpu)
    else:
        tlx.set_device('CPU')

    main(args)
