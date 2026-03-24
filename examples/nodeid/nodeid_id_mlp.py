# !/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import argparse
import random
import numpy as np
import sys
from pathlib import Path
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss

os.environ.setdefault('TL_BACKEND', 'torch')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gammagl.datasets import Planetoid
from gammagl.utils import mask_to_index
from gammagl.models import MLP


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
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss


def evaluate(model, data, metrics):
    model.set_eval()
    logits = model(data['x'])

    train_logits = tlx.gather(logits, data['train_idx'])
    train_y = tlx.gather(data['y'], data['train_idx'])
    valid_logits = tlx.gather(logits, data['valid_idx'])
    valid_y = tlx.gather(data['y'], data['valid_idx'])
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])

    result = {
        'train_acc': calculate_acc(train_logits, train_y, metrics),
        'valid_acc': calculate_acc(valid_logits, valid_y, metrics),
        'test_acc': calculate_acc(test_logits, test_y, metrics),
    }
    return result


def load_semantic_id(path):
    data = np.load(path)
    if 'arr_0' in data:
        arr = data['arr_0']
    else:
        first_key = list(data.keys())[0]
        arr = data[first_key]
    return tlx.convert_to_tensor(arr, dtype=tlx.float32)


def main(args):
    split_seed = args.seed if args.split_seed < 0 else args.split_seed
    train_seed = args.seed if args.train_seed < 0 else args.train_seed

    fix_seed(train_seed)

    if str.lower(args.dataset) not in ['cora', 'citeseer', 'pubmed']:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]

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

    semantic_path = args.semantic_id_path or f'semantic_ID_{args.dataset}.npz'
    semantic_id = load_semantic_id(semantic_path)
    semantic_shape = tlx.get_tensor_shape(semantic_id)
    if semantic_shape[0] != graph.num_nodes:
        raise ValueError(
            f'semantic id node count mismatch: got {semantic_shape[0]}, expected {graph.num_nodes}'
        )

    feats = semantic_id
    y = graph.y
    if len(tlx.get_tensor_shape(y)) > 1:
        y = tlx.squeeze(y, axis=-1)

    input_dim = semantic_shape[1] if args.num_id <= 0 else args.num_id
    if input_dim != semantic_shape[1]:
        raise ValueError(f'num_id={args.num_id} does not match semantic dim={semantic_shape[1]}')

    if args.norm_type == 'layer':
        raise ValueError('GammaGL MLP does not support layer norm in this script, please use --norm_type none or batch')

    norm = tlx.nn.BatchNorm1d(gamma_init='ones') if args.norm_type == 'batch' else None
    model = MLP(
        in_channels=input_dim,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        norm=norm,
        plain_last=True,
    )

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = model.trainable_weights

    loss_func = SemiSpvzLoss(model, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        'x': feats,
        'y': y,
        'train_idx': split_idx['train'],
        'valid_idx': split_idx['valid'],
        'test_idx': split_idx['test'],
    }

    best_valid = float('-inf')
    best_test = float('-inf')

    print(f'dataset={args.dataset}, stage=ID-MLP, semantic_id={semantic_path}')

    for epoch in range(args.n_epoch):
        model.set_train()
        train_loss = train_one_step(data, y)

        result = evaluate(model, data, metrics)
        if result['valid_acc'] > best_valid:
            best_valid = result['valid_acc']
            best_test = result['test_acc']

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NodeID Stage-2 ID-MLP in GammaGL')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_seed', type=int, default=-1)
    parser.add_argument('--train_seed', type=int, default=-1)

    parser.add_argument('--rand_split', action='store_true')
    parser.add_argument('--train_prop', type=float, default=0.6)
    parser.add_argument('--valid_prop', type=float, default=0.2)

    parser.add_argument('--semantic_id_path', type=str, default='')
    parser.add_argument('--num_id', type=int, default=-1)

    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--display_step', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_coef', type=float, default=5e-4)

    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--norm_type', type=str, default='none', choices=['none', 'batch', 'layer'])

    args = parser.parse_args()

    if args.gpu >= 0:
        tlx.set_device('GPU', args.gpu)
    else:
        tlx.set_device('CPU')

    main(args)