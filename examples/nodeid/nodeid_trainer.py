# !/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import argparse
import random
import numpy as np
import sys
from pathlib import Path

os.environ.setdefault('TL_BACKEND', 'torch')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss

from gammagl.datasets import Planetoid
from gammagl.utils import add_self_loops, remove_self_loops, to_undirected, mask_to_index
from gammagl.models import NodeIDGNN


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if os.environ.get('TL_BACKEND', 'torch').lower() == 'torch':
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        except Exception:
            pass


def random_split_indices(num_nodes, train_prop=0.6, valid_prop=0.2, split_seed=None):
    if split_seed is None:
        perm = np.random.permutation(num_nodes)
    else:
        rng = np.random.RandomState(split_seed)
        perm = rng.permutation(num_nodes)
    train_num = int(train_prop * num_nodes)
    valid_num = int(valid_prop * num_nodes)
    train_idx = tlx.convert_to_tensor(perm[:train_num], dtype=tlx.int64)
    valid_idx = tlx.convert_to_tensor(perm[train_num:train_num + valid_num], dtype=tlx.int64)
    test_idx = tlx.convert_to_tensor(perm[train_num + valid_num:], dtype=tlx.int64)
    return {
        'train': train_idx,
        'valid': valid_idx,
        'test': test_idx,
    }


def calculate_acc(logits, y, metrics):
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn, commit_weight):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.commit_weight = commit_weight

    def forward(self, data, y):
        logits, commit_loss, _, _ = self.backbone_network(data['x'], data['edge_index'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y) + self.commit_weight * commit_loss
        return loss


def evaluate(model, data, metrics):
    model.set_eval()
    logits, _, _, _ = model(data['x'], data['edge_index'])

    train_logits = tlx.gather(logits, data['train_idx'])
    train_y = tlx.gather(data['y'], data['train_idx'])
    valid_logits = tlx.gather(logits, data['valid_idx'])
    valid_y = tlx.gather(data['y'], data['valid_idx'])
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])

    return {
        'train_acc': calculate_acc(train_logits, train_y, metrics),
        'valid_acc': calculate_acc(valid_logits, valid_y, metrics),
        'test_acc': calculate_acc(test_logits, test_y, metrics),
    }


def save_semantic_id(model, x, edge_index, output_path):
    model.set_eval()
    _, _, id_list_concat, _ = model(x, edge_index)
    np.savez(output_path, tlx.convert_to_numpy(id_list_concat))
    print(f'saved semantic id to {output_path}')


def extract_semantic_id(model, x, edge_index):
    model.set_eval()
    _, _, id_list_concat, _ = model(x, edge_index)
    return tlx.convert_to_numpy(id_list_concat)


def main(args):
    split_seed = args.seed if args.split_seed < 0 else args.split_seed
    train_seed = args.seed if args.train_seed < 0 else args.train_seed

    fix_seed(train_seed)

    if str.lower(args.dataset) not in ['cora', 'citeseer', 'pubmed']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]

    edge_index = to_undirected(graph.edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=graph.num_nodes, n_loops=1)

    if args.rand_split:
        split_idx = random_split_indices(
            graph.num_nodes,
            train_prop=args.train_prop,
            valid_prop=args.valid_prop,
            split_seed=split_seed,
        )
    else:
        split_idx = {
            'train': mask_to_index(graph.train_mask),
            'valid': mask_to_index(graph.val_mask),
            'test': mask_to_index(graph.test_mask),
        }

    y = graph.y
    if len(tlx.get_tensor_shape(y)) > 1:
        y = tlx.squeeze(y, axis=-1)

    model = NodeIDGNN(
        in_channels=dataset.num_node_features,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        local_layers=args.local_layers,
        in_dropout=args.in_dropout,
        dropout=args.dropout,
        heads=args.num_heads,
        pre_ln=args.pre_ln,
        kmeans=args.kmeans,
        num_codes=args.num_codes,
        gnn=args.method,
    )

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = model.trainable_weights

    loss_func = SemiSpvzLoss(model, tlx.losses.softmax_cross_entropy_with_logits, args.commit_weight)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        'x': graph.x,
        'y': y,
        'edge_index': edge_index,
        'train_idx': split_idx['train'],
        'valid_idx': split_idx['valid'],
        'test_idx': split_idx['test'],
    }

    best_valid = 0.0
    best_test = 0.0
    best_semantic_id = None

    print(f'dataset={args.dataset}, method={args.method}')

    for epoch in range(args.n_epoch):
        model.set_train()
        train_loss = train_one_step(data, y)

        result = evaluate(model, data, metrics)

        if result['valid_acc'] > best_valid:
            best_valid = result['valid_acc']
            best_test = result['test_acc']
            if args.save_semantic_id:
                best_semantic_id = extract_semantic_id(model, data['x'], data['edge_index'])

        if epoch % args.display_step == 0:
            print(
                f"Epoch [{epoch + 1:0>3d}]  "
                f"train loss: {train_loss.item():.4f}  "
                f"train acc: {result['train_acc'] * 100:.2f}%  "
                f"val acc: {result['valid_acc'] * 100:.2f}%  "
                f"test acc: {result['test_acc'] * 100:.2f}%  "
                f"best val: {best_valid * 100:.2f}%  "
                f"best test: {best_test * 100:.2f}%"
            )

    if args.save_semantic_id:
        semantic_id_path = args.semantic_id_path or f'semantic_ID_{args.dataset}.npz'
        if best_semantic_id is None:
            save_semantic_id(model, data['x'], data['edge_index'], semantic_id_path)
        else:
            np.savez(semantic_id_path, best_semantic_id)
            print(f'saved semantic id to {semantic_id_path} (best-valid checkpoint)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NodeID Stage-1 (GNN + VQ) in GammaGL style')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_seed', type=int, default=-1)
    parser.add_argument('--train_seed', type=int, default=-1)
    parser.add_argument('--rand_split', action='store_true')
    parser.add_argument('--train_prop', type=float, default=0.6)
    parser.add_argument('--valid_prop', type=float, default=0.2)

    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--display_step', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_coef', type=float, default=5e-4)

    parser.add_argument('--method', type=str, default='gat', choices=['gat', 'gcn'])
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=7)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--in_dropout', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pre_ln', action='store_true')

    parser.add_argument('--kmeans', type=int, default=1)
    parser.add_argument('--num_codes', type=int, default=16)
    parser.add_argument('--commit_weight', type=float, default=1.0)
    parser.add_argument('--save_semantic_id', action='store_true')
    parser.add_argument('--semantic_id_path', type=str, default='')

    args = parser.parse_args()

    if args.gpu >= 0:
        tlx.set_device('GPU', args.gpu)
    else:
        tlx.set_device('CPU')

    main(args)
