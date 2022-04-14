# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/20 13:47
# @Author  : clear
# @FileName: trainer.py.py
import os

os.environ['TL_BACKEND'] = 'paddle'  # set your backend here, default `tensorflow`
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
import argparse
import numpy as np
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import RGCN
from gammagl.utils.loop import add_self_loops
from tensorlayerx.model import TrainOneStep, WithLoss
import gammagl.transforms as T
from gammagl.datasets import IMDB
import os.path as osp

def onehot(size, idx):
    diag1 = tlx.convert_to_tensor(np.eye(size), tlx.float32)
    tlx.gather(diag1, idx)

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self._backbone(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
        if tlx.BACKEND == 'mindspore':
            idx = tlx.convert_to_tensor([i for i, v in enumerate(data['train_mask']) if v], dtype=tlx.int64)
            train_logits = tlx.gather(logits, idx)
            train_label = tlx.gather(label, idx)
        else:
            train_logits = logits[data['train_mask']]
            train_label = label[data['train_mask']]
        loss = self._loss_fn(train_logits, train_label)
        return loss

def evaluate(net, data, y, mask, metrics):
    net.set_eval()
    logits = net(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
    if tlx.BACKEND == 'mindspore':
        idx = tlx.convert_to_tensor([i for i, v in enumerate(mask) if v],dtype=tlx.int64)
        _logits = tlx.gather(logits,idx)
        _label = tlx.gather(y,idx)
    else:
        _logits = logits[mask]
        _label = y[mask]
    metrics.update(_logits, _label)
    acc = metrics.result()
    metrics.reset()
    return acc


def main(args):
    if str.lower(args.dataset) not in ['cora','pubmed','citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/IMDB')
    metapaths = [[('movie', 'actor'), ('actor', 'movie')],
                 [('movie', 'director'), ('director', 'movie')]]
    transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edges=True,
                               drop_unconnected_nodes=True)
    dataset = IMDB(path, transform=transform)
    graph = dataset[0]

    net = RGCN(feature_dim=graph.num_nodes,
               hidden_dim=args.hidden_dim,
               num_class=graph.y.shape[1],
               num_relations=len(metapaths),
               name="RGCN")

    optimizer = tlx.optimizers.Adam(learning_rate=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=16, help="dimention of hidden layers")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument('--num_bases', type=int, default=None, help='number of bases')
    parser.add_argument('--num_blocks', type=int, default=None, help='numbere of blocks')
    parser.add_argument("--aggregation", type=str, default='sum', help='aggregate type')
    parser.add_argument('--dataset', type=str, default='ACM', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    args = parser.parse_args()

    main(args)
