# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   coed_trainer.py
@Time    :   2024/12/30 15:30:00
@Author  :   GammaGL
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR

import argparse
import numpy as np
import tensorlayerx as tlx
from gammagl.datasets import WebKB, WikipediaNetwork
from gammagl.models import CoEDModel
from gammagl.utils import mask_to_index
from tensorlayerx.model import TrainOneStep, WithLoss
import gammagl.transforms as T

from geom_planetoid import load_planetoid_with_geom_splits


class SemiSpvzLoss(WithLoss):
    r"""Loss wrapper for semi-supervised node classification."""

    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss


def calculate_acc(logits, y, metrics):
    r"""Compute accuracy via the TLX metrics API."""
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


def get_edge_index_and_theta(edge_index):
    r"""Build the fuzzy edge list and initial phase angles from an edge_index.

    Symmetric (undirected) edges are kept only once with theta = pi/4;
    directed edges are kept as-is with theta = 0.
    """
    src = tlx.convert_to_numpy(edge_index[0]).tolist()
    dst = tlx.convert_to_numpy(edge_index[1]).tolist()

    edges = [(int(u), int(v)) for u, v in zip(src, dst) if u != v]
    edge_set = set(edges)

    triu_symm_edges = []
    triu_dir_edges = []
    tril_dir_edges = []

    for u, v in edges:
        if u < v:
            if (v, u) in edge_set:
                triu_symm_edges.append((u, v))
            else:
                triu_dir_edges.append((u, v))
        elif u > v and (v, u) not in edge_set:
            tril_dir_edges.append((u, v))

    triu_symm_edges = sorted(set(triu_symm_edges))
    triu_dir_edges = sorted(set(triu_dir_edges))
    tril_dir_edges = sorted(set(tril_dir_edges))

    if triu_symm_edges:
        if not triu_dir_edges and not tril_dir_edges:
            processed_edges = triu_symm_edges
            theta = [np.pi / 4.0] * len(triu_symm_edges)
        else:
            processed_edges = triu_dir_edges + tril_dir_edges + triu_symm_edges
            theta = [0.0] * (len(triu_dir_edges) + len(tril_dir_edges)) + [np.pi / 4.0] * len(triu_symm_edges)
    else:
        processed_edges = triu_dir_edges + tril_dir_edges
        theta = [0.0] * len(processed_edges)

    edge_index_fuzzy = tlx.convert_to_tensor(np.array(processed_edges, dtype=np.int64).T, dtype=tlx.int64)
    theta = tlx.convert_to_tensor(np.array(theta, dtype=np.float32), dtype=tlx.float32)
    return edge_index_fuzzy, theta


def get_fuzzy_laplacian(edge_index, theta, num_nodes, edge_weight=None, add_self_loop=False):
    r"""Construct normalized directional edge weights for CoED message passing.

    This implements the fuzzy Laplacian normalization described in the paper.
    For each edge (i, j) with phase angle theta_k, the directional weights are:
      - src-to-dst: cos^2(theta_k)
      - dst-to-src: sin^2(theta_k)
    These are then symmetrically normalized by node degrees.
    """
    from gammagl.mpops import unsorted_segment_sum

    senders = edge_index[0]
    receivers = edge_index[1]

    if edge_weight is None:
        edge_weight = tlx.ones((tlx.get_tensor_shape(senders)[0],), dtype=tlx.float32)

    theta = tlx.cast(theta, tlx.float32)
    edge_weight = tlx.cast(edge_weight, tlx.float32)
    cos_sq = tlx.cos(theta) ** 2
    sin_sq = tlx.sin(theta) ** 2

    conv_senders = tlx.concat([senders, receivers], axis=0)
    conv_receivers = tlx.concat([receivers, senders], axis=0)
    out_weight = tlx.concat([cos_sq * edge_weight, sin_sq * edge_weight], axis=0)
    in_weight = tlx.concat([sin_sq * edge_weight, cos_sq * edge_weight], axis=0)

    if add_self_loop:
        self_loops = tlx.arange(start=0, limit=num_nodes, dtype=tlx.int64)
        ones = tlx.ones((num_nodes,), dtype=tlx.float32)
        conv_senders = tlx.concat([conv_senders, self_loops], axis=0)
        conv_receivers = tlx.concat([conv_receivers, self_loops], axis=0)
        out_weight = tlx.concat([out_weight, ones], axis=0)
        in_weight = tlx.concat([in_weight, ones], axis=0)

    deg_senders = tlx.reshape(
        unsorted_segment_sum(out_weight, conv_senders, num_segments=num_nodes), (-1,)
    ) + 1e-12
    deg_receivers = tlx.reshape(
        unsorted_segment_sum(in_weight, conv_senders, num_segments=num_nodes), (-1,)
    ) + 1e-12

    deg_inv_sqrt_senders = tlx.where(
        deg_senders < 1e-11, tlx.zeros_like(deg_senders), tlx.pow(deg_senders, -0.5)
    )
    deg_inv_sqrt_receivers = tlx.where(
        deg_receivers < 1e-11, tlx.zeros_like(deg_receivers), tlx.pow(deg_receivers, -0.5)
    )

    ew_src_to_dst = (
        tlx.gather(deg_inv_sqrt_senders, conv_senders)
        * out_weight
        * tlx.gather(deg_inv_sqrt_receivers, conv_receivers)
    )
    ew_dst_to_src = (
        tlx.gather(deg_inv_sqrt_receivers, conv_senders)
        * in_weight
        * tlx.gather(deg_inv_sqrt_senders, conv_receivers)
    )

    conv_edge_index = tlx.stack([conv_senders, conv_receivers], axis=0)
    conv_edge_weight = (tlx.reshape(ew_src_to_dst, (-1, 1)), tlx.reshape(ew_dst_to_src, (-1, 1)))
    return conv_edge_index, conv_edge_weight


def set_seed(seed):
    r"""Set random seeds for reproducible runs."""
    np.random.seed(seed)
    tlx.set_seed(seed)


def main(args):
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    dataset_name = str.lower(args.dataset)

    if dataset_name in ['cora', 'pubmed', 'citeseer']:
        # Planetoid with Geom-GCN 10 fixed splits
        dataset, graph = load_planetoid_with_geom_splits(
            root=args.dataset_path, name=dataset_name,
            num_splits=args.num_splits, transform=T.NormalizeFeatures(),
        )
    elif dataset_name in ['texas', 'wisconsin', 'cornell']:
        dataset = WebKB(args.dataset_path, dataset_name, transform=T.NormalizeFeatures())
        graph = dataset[0]
        # WebKB masks are flat 1D: concatenation of 10 splits
        n = graph.num_nodes
        train_idx = mask_to_index(graph.train_mask[args.split_idx * n: (args.split_idx + 1) * n])
        val_idx = mask_to_index(graph.val_mask[args.split_idx * n: (args.split_idx + 1) * n])
        test_idx = mask_to_index(graph.test_mask[args.split_idx * n: (args.split_idx + 1) * n])
    elif dataset_name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(args.dataset_path, dataset_name, geom_gcn_preprocess=True)
        graph = dataset[0]
        # WikipediaNetwork masks are flat 1D: concatenation of 10 splits
        n = graph.num_nodes
        train_idx = mask_to_index(graph.train_mask[args.split_idx * n: (args.split_idx + 1) * n])
        val_idx = mask_to_index(graph.val_mask[args.split_idx * n: (args.split_idx + 1) * n])
        test_idx = mask_to_index(graph.test_mask[args.split_idx * n: (args.split_idx + 1) * n])
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    # ------------------------------------------------------------------
    # 2. Build fuzzy edge structure (dataset-level, shared across splits)
    # ------------------------------------------------------------------
    if args.remove_existing_self_loop:
        # Remove self-loops from the original edge_index
        src = tlx.convert_to_numpy(graph.edge_index[0])
        dst = tlx.convert_to_numpy(graph.edge_index[1])
        mask = src != dst
        graph.edge_index = tlx.convert_to_tensor(
            np.array([src[mask], dst[mask]], dtype=np.int64), dtype=tlx.int64
        )

    edge_index, theta = get_edge_index_and_theta(graph.edge_index)
    num_nodes = graph.num_nodes

    conv_edge_index, conv_edge_weight = get_fuzzy_laplacian(
        edge_index=edge_index,
        theta=theta,
        num_nodes=num_nodes,
        add_self_loop=args.self_loop,
    )

    # ------------------------------------------------------------------
    # 3. Run multi-split evaluation
    # ------------------------------------------------------------------
    split_test_accs = []

    for split_id in range(args.num_splits):
        # Reload masks for this split
        if dataset_name in ['cora', 'pubmed', 'citeseer']:
            # Geom-GCN splits: 2D masks [num_nodes, num_splits]
            train_idx = mask_to_index(graph.train_mask[:, split_id])
            val_idx = mask_to_index(graph.val_mask[:, split_id])
            test_idx = mask_to_index(graph.test_mask[:, split_id])
        else:
            n = graph.num_nodes
            train_idx = mask_to_index(graph.train_mask[split_id * n: (split_id + 1) * n])
            val_idx = mask_to_index(graph.val_mask[split_id * n: (split_id + 1) * n])
            test_idx = mask_to_index(graph.test_mask[split_id * n: (split_id + 1) * n])

        data = {
            "x": graph.x,
            "y": graph.y,
            "edge_index": conv_edge_index,
            "edge_weight": conv_edge_weight,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "val_idx": val_idx,
            "num_nodes": num_nodes,
        }

        for run in range(args.runs):
            set_seed(args.seed + split_id * 97 + run)

            # Instantiate model
            jk = args.jumping_knowledge if args.jumping_knowledge != "None" else None
            net = CoEDModel(
                feature_dim=dataset.num_node_features,
                hidden_dim=args.hidden_dim,
                num_class=dataset.num_classes,
                num_layers=args.num_layers,
                alpha=args.alpha,
                drop_rate=args.drop_rate,
                normalize=args.normalize,
                self_feature_transform=args.self_feature_transform,
                jumping_knowledge=jk,
                name="CoED",
            )

            optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
            metrics = tlx.metrics.Accuracy()
            train_weights = net.trainable_weights

            loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
            train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

            best_val_acc = 0
            best_test_acc = 0
            bad_counter = 0

            for epoch in range(1, args.n_epoch + 1):
                net.set_train()
                train_loss = train_one_step(data, graph.y)

                net.set_eval()
                logits = net(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])

                val_logits = tlx.gather(logits, data['val_idx'])
                val_y = tlx.gather(data['y'], data['val_idx'])
                val_acc = calculate_acc(val_logits, val_y, metrics)

                test_logits = tlx.gather(logits, data['test_idx'])
                test_y = tlx.gather(data['y'], data['test_idx'])
                test_acc = calculate_acc(test_logits, test_y, metrics)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    bad_counter = 0
                    net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')
                else:
                    bad_counter += 1

                if epoch % args.print_freq == 0 or epoch == 1:
                    print(
                        "split {:02d} run {:02d} epoch {:04d} "
                        "loss {:.4f} val {:.4f} best_test {:.4f} patience {}/{}".format(
                            split_id, run, epoch,
                            float(train_loss.item()),
                            val_acc, best_test_acc,
                            bad_counter, args.patience,
                        )
                    )

                if bad_counter >= args.patience:
                    break

            # Restore best model for final evaluation
            net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict')
            if tlx.BACKEND == 'torch':
                net.to(data['x'].device)
            net.set_eval()
            logits = net(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
            test_logits = tlx.gather(logits, data['test_idx'])
            test_y = tlx.gather(data['y'], data['test_idx'])
            best_test_acc = calculate_acc(test_logits, test_y, metrics)
            split_test_accs.append(best_test_acc)
            print("split {:02d} run {:02d} best test acc: {:.5f}".format(split_id, run, best_test_acc * 100.0))

    mean_test = float(np.mean(split_test_accs) * 100.0)
    std_test = float(np.std(split_test_accs) * 100.0)
    print("test acc: {:.5f} +/- {:.5f}".format(mean_test, std_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CoED-GNN node classification with GammaGL/TensorLayerX.")

    # Dataset
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'texas', 'wisconsin', 'chameleon', 'squirrel'],
                        help='Dataset name.')
    parser.add_argument('--dataset_path', type=str, default=r'', help='Path to save/load dataset.')
    parser.add_argument('--num_splits', type=int, default=10, help='Number of fixed splits to evaluate.')
    parser.add_argument('--split_idx', type=int, default=0, help='Unused when num_splits > 0.')
    parser.add_argument('--runs', type=int, default=1, help='Runs per split.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Direction convex combination parameter.')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='Feature dropout rate.')
    parser.add_argument('--normalize', dest='normalize', action='store_true',
                        help='L2-normalize hidden features at each layer.')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.add_argument('--self_feature_transform', dest='self_feature_transform', action='store_true',
                        help='Learn a separate self-feature transform branch.')
    parser.add_argument('--no_self_feature_transform', dest='self_feature_transform', action='store_false')
    parser.add_argument('--self_loop', dest='self_loop', action='store_true',
                        help='Mix self features into directional messages.')
    parser.add_argument('--no_self_loop', dest='self_loop', action='store_false')
    parser.add_argument('--jumping_knowledge', type=str, default='None',
                        choices=['None', 'cat', 'max', 'lstm'],
                        help='Jumping-knowledge aggregation type.')
    parser.add_argument('--remove_existing_self_loop', dest='remove_existing_self_loop',
                        action='store_true',
                        help='Remove existing self-loops from the graph before processing.')
    parser.add_argument('--no_remove_existing_self_loop', dest='remove_existing_self_loop',
                        action='store_false')

    # Training
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--l2_coef', type=float, default=0.0, help='Weight decay (L2 regularization).')
    parser.add_argument('--n_epoch', type=int, default=5000, help='Max training epochs.')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience.')
    parser.add_argument('--print_freq', type=int, default=50, help='Print frequency (epochs).')

    # System
    parser.add_argument('--best_model_path', type=str, default=r'./', help='Path to save best model.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index, -1 for CPU.')

    parser.set_defaults(
        normalize=False,
        self_feature_transform=False,
        self_loop=True,
        remove_existing_self_loop=False,
    )

    args = parser.parse_args()

    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    main(args)
