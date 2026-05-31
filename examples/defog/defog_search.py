#!/usr/bin/env python3
"""Hyperparameter grid search for DeFoG sampling parameters.

Matches the original DeFoG ``search_hyperparameters()`` API:

* ``search_distortion``   -- time-distortion schedule
* ``search_stochasticity``-- eta (DB rate strength)
* ``search_target_guidance``-- omega (target-guidance strength)

Results are written as CSV files under ``--save_dir``.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorlayerx as tlx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gammagl.models.defog import DeFoGModel
from extra_features import ExtraFeatures, DummyExtraFeatures
from extra_features_molecular import ExtraMolecularFeatures, DummyMolecularFeatures
from noise_distribution import NoiseDistribution
from rate_matrix import RateMatrixDesigner
from time_distorter import TimeDistorter
from sampler import sample_batch
from evaluator import evaluate_generated_graphs
from defog_trainer import (
    load_model_snapshot_for_sampling
)
from dataset_utils import load_real_dataset


def _build_model(args, num_node_types, num_edge_types, test_labels):
    """Instantiate model with correct I/O dims."""
    dx = num_node_types
    de = num_edge_types
    dy = test_labels.shape[-1] if test_labels is not None else 0

    return DeFoGModel(
        n_layers=args.n_layers,
        hidden_mlp_X=args.hidden_mlp_X,
        hidden_mlp_E=args.hidden_mlp_E,
        hidden_mlp_y=args.hidden_mlp_y,
        dx=dx,
        de=de,
        dy=dy,
        n_head=args.n_head,
        dim_ffX=args.dim_ffX,
        dim_ffE=args.dim_ffE,
        dim_ffy=args.dim_ffy,
    )


def _load_data_and_model(args):
    """Load dataset, build model, and load checkpoint."""
    remove_h = args.remove_h
    if remove_h is None and args.dataset == 'qm9':
        remove_h = True

    print(f"[search] Loading dataset '{args.dataset}' (remove_h={remove_h})...")
    train_graphs, val_ds, test_ds, dataset_infos, num_node_types, num_edge_types, test_labels = (
        load_real_dataset(
            args.dataset,
            remove_h=remove_h,
            conditional=args.conditional,
            target=args.target,
        )
    )

    print(f"[search] Building model...")
    model = _build_model(args, num_node_types, num_edge_types, test_labels)

    print(f"[search] Loading checkpoint from {args.save_dir}...")
    load_model_snapshot_for_sampling(model, args.save_dir, ema_decay=args.ema_decay)

    # Helpers
    noise_dist = NoiseDistribution(transition=args.transition, limit_dist=dataset_infos)
    rate_matrix_designer = RateMatrixDesigner(
        rdb=args.rdb,
        rdb_crit=args.rdb_crit,
        eta=args.eta,
        omega=args.omega,
        limit_dist=dataset_infos,
    )
    time_distorter = TimeDistorter(
        train_distortion='identity',
        sample_distortion=args.sample_distortion,
    )

    if args.extra_features == 'rrwp':
        extra_features = ExtraFeatures(args.rrwp_steps, dataset_infos)
    else:
        extra_features = DummyExtraFeatures()

    if args.dataset in ('qm9', 'guacamol', 'zinc250k', 'moses'):
        domain_features = ExtraMolecularFeatures(dataset_infos)
    else:
        domain_features = DummyExtraFeatures()

    node_dist = dataset_infos['node_dist'].astype(np.float32)

    config = {
        'model': model,
        'noise_dist': noise_dist,
        'rate_matrix_designer': rate_matrix_designer,
        'time_distorter': time_distorter,
        'extra_features': extra_features,
        'domain_features': domain_features,
        'node_dist': node_dist,
        'dataset_name': args.dataset,
        'graphs': train_graphs,
        'test_ds': test_ds,
        'dataset_infos': dataset_infos,
        'num_node_types': num_node_types,
    }
    return config


def _run_once(config, sample_steps, batch_size, **sample_kwargs):
    """Sample and evaluate once with the current config."""
    t0 = time.time()
    generated = sample_batch(
        config['model'],
        config['noise_dist'],
        config['rate_matrix_designer'],
        config['time_distorter'],
        config['extra_features'],
        config['domain_features'],
        config['node_dist'],
        sample_steps=sample_steps,
        batch_size=batch_size,
        **sample_kwargs
    )
    t_sample = time.time() - t0

    t0 = time.time()
    metrics = evaluate_generated_graphs(
        generated,
        config['dataset_name'],
        config['graphs'],
        config['test_ds'],
        config['dataset_infos'],
        config['num_node_types'],
    )
    t_eval = time.time() - t0

    metrics['_sample_time'] = round(t_sample, 2)
    metrics['_eval_time'] = round(t_eval, 2)
    return metrics


def _flatten_metrics(metrics):
    """Keep only scalar / JSON-serialisable entries for CSV."""
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            flat[k] = float(v)
        elif isinstance(v, (list, dict, tuple)):
            flat[k] = str(v)
        elif v is None:
            flat[k] = None
        else:
            flat[k] = v
    return flat


def search_distortion(config, args):
    """Grid search over time-distortion schedule."""
    results = []
    csv_path = os.path.join(args.save_dir, 'search_distortion.csv')
    distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]

    for num_step in args.num_step_list:
        for distortion in distortion_list:
            print(f"\n{'#' * 60}")
            print(f"# steps={num_step}, distortion={distortion}")
            print(f"{'#' * 60}")

            config['time_distorter'] = TimeDistorter(
                train_distortion='identity',
                sample_distortion=distortion,
            )

            metrics = _run_once(config, num_step, args.num_search_samples)
            row = _flatten_metrics(metrics)
            row['num_step'] = num_step
            row['distortion'] = distortion
            results.append(row)

            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)
            print(df.to_string(index=False))

    # Reset to default
    config['time_distorter'] = TimeDistorter(
        train_distortion='identity',
        sample_distortion='identity',
    )
    return df


def search_stochasticity(config, args):
    """Grid search over eta (DB stochasticity)."""
    results = []
    csv_path = os.path.join(args.save_dir, 'search_stochasticity.csv')
    eta_list = [0.0, 5, 10, 25, 50, 100, 200, 300, 500]

    for num_step in args.num_step_list:
        for eta in eta_list:
            print(f"\n{'#' * 60}")
            print(f"# steps={num_step}, eta={eta}")
            print(f"{'#' * 60}")

            config['rate_matrix_designer'].eta = float(eta)

            metrics = _run_once(config, num_step, args.num_search_samples)
            row = _flatten_metrics(metrics)
            row['num_step'] = num_step
            row['eta'] = eta
            results.append(row)

            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)
            print(df.to_string(index=False))

    # Reset
    config['rate_matrix_designer'].eta = 0.0
    return df


def search_target_guidance(config, args):
    """Grid search over omega (target guidance)."""
    results = []
    csv_path = os.path.join(args.save_dir, 'search_target_guidance.csv')
    omega_list = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0]

    for num_step in args.num_step_list:
        for omega in omega_list:
            print(f"\n{'#' * 60}")
            print(f"# steps={num_step}, omega={omega}")
            print(f"{'#' * 60}")

            config['rate_matrix_designer'].omega = float(omega)

            metrics = _run_once(config, num_step, args.num_search_samples)
            row = _flatten_metrics(metrics)
            row['num_step'] = num_step
            row['omega'] = omega
            results.append(row)

            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)
            print(df.to_string(index=False))

    # Reset
    config['rate_matrix_designer'].omega = 0.0
    return df


def main():
    parser = argparse.ArgumentParser(description='DeFoG Sampling Hyperparameter Search')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['synthetic', 'planar', 'tree', 'sbm', 'comm20',
                                 'qm9', 'guacamol', 'zinc250k', 'moses', 'tls'])
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Checkpoint directory (contains best_model.npz or last_model.npz)')
    parser.add_argument('--remove_h', action='store_true')
    parser.add_argument('--with_h', action='store_true')
    parser.add_argument('--conditional', action='store_true')
    parser.add_argument('--target', type=str, default='mu')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--search_mode', type=str, default='all',
                        choices=['all', 'distortion', 'stochasticity', 'target_guidance'])
    parser.add_argument('--num_search_samples', type=int, default=512,
                        help='Number of graphs to generate per search point')
    parser.add_argument('--num_step_list', type=int, nargs='+', default=None,
                        help='Override default step list')

    # Model / sampling defaults
    parser.add_argument('--n_layers', type=int, default=9)
    parser.add_argument('--hidden_mlp_X', type=int, default=256)
    parser.add_argument('--hidden_mlp_E', type=int, default=256)
    parser.add_argument('--hidden_mlp_y', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--dim_ffX', type=int, default=256)
    parser.add_argument('--dim_ffE', type=int, default=128)
    parser.add_argument('--dim_ffy', type=int, default=128)
    parser.add_argument('--ema_decay', type=float, default=0.0)
    parser.add_argument('--transition', type=str, default='marginal')
    parser.add_argument('--extra_features', type=str, default='rrwp')
    parser.add_argument('--rrwp_steps', type=int, default=12)
    parser.add_argument('--rdb', type=str, default='general')
    parser.add_argument('--rdb_crit', type=str, default='max_marginal')
    parser.add_argument('--sample_distortion', type=str, default='identity')
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--omega', type=float, default=0.0)

    args = parser.parse_args()

    # Resolve remove_h
    if args.remove_h:
        args.remove_h = True
    elif args.with_h:
        args.remove_h = False
    else:
        args.remove_h = True if args.dataset == 'qm9' else None

    # Default step lists per dataset (matching original DeFoG)
    if args.num_step_list is None:
        if args.dataset == 'qm9':
            args.num_step_list = [1, 5, 10, 50, 100, 500]
        elif args.dataset in ('guacamol', 'moses', 'zinc250k'):
            args.num_step_list = [50]
        else:
            args.num_step_list = [5, 10, 50, 100, 1000]

    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        if tlx.BACKEND == 'torch':
            import torch
            torch.cuda.set_device(args.gpu)

    print(f"[search] Mode: {args.search_mode}")
    print(f"[search] Dataset: {args.dataset}")
    print(f"[search] Step list: {args.num_step_list}")
    print(f"[search] Samples per point: {args.num_search_samples}")
    print(f"[search] Save dir: {args.save_dir}")

    config = _load_data_and_model(args)

    if args.search_mode == 'all':
        search_distortion(config, args)
        search_stochasticity(config, args)
        search_target_guidance(config, args)
    elif args.search_mode == 'distortion':
        search_distortion(config, args)
    elif args.search_mode == 'stochasticity':
        search_stochasticity(config, args)
    elif args.search_mode == 'target_guidance':
        search_target_guidance(config, args)

    print("\n[search] Finished. Results saved to:")
    if args.search_mode in ('all', 'distortion'):
        print(f"  {os.path.join(args.save_dir, 'search_distortion.csv')}")
    if args.search_mode in ('all', 'stochasticity'):
        print(f"  {os.path.join(args.save_dir, 'search_stochasticity.csv')}")
    if args.search_mode in ('all', 'target_guidance'):
        print(f"  {os.path.join(args.save_dir, 'search_target_guidance.csv')}")


if __name__ == '__main__':
    main()
