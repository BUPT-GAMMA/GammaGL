# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/20 13:47
# @Author  : hanhui
# @FileName: trainer.py.py
import os
os.environ['TL_BACKEND'] = 'torch'  # set your backend here, default `tensorflow`
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
from gammagl.models import RGCN
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.datasets import Entities
import os.path as osp
import numpy as np


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self._backbone(data['edge_index'], data['edge_type'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = y # tlx.gather(y, data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss


def evaluate(net, data, y, metrics):
    net.set_eval()
    logits = net(data['edge_index'], data['edge_type'])
    _logits = tlx.gather(logits, data['test_idx'])
    _y = y # tlx.gather(y, data['test_idx'])
    metrics.update(_logits, _y)
    acc = metrics.result()
    metrics.reset()
    return acc

# not use , perform on full graph
def k_hop_subgraph(node_idx, num_hops, edge_index, num_nodes, relabel_nodes=False, flow='source_to_target'):
    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = np.zeros(num_nodes, dtype=np.bool)
    edge_mask = np.zeros(row.shape[0], dtype=np.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = node_idx.flatten()
    else:
        node_idx = node_idx
    subsets = [node_idx]
    for _ in range(num_hops):
        node_mask.fill(False)
        node_mask[subsets[-1]] = True
        edge_mask = node_mask[row]
        subsets.append(col[edge_mask])

    subset, inv = np.unique(np.concatenate(subsets), return_inverse=True)
    numel = 1
    for n in node_idx.shape:
        numel *= n
    inv = inv[:numel]

    node_mask.fill(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = -np.ones((num_nodes, ))
        node_idx[subset] = np.arange(subset.shape[0])
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


def main(args):
    # load cora dataset
    if str.lower(args.dataset) not in ['aifb', 'mutag', 'bgs', 'am']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../Entities')
    dataset = Entities(path, args.dataset)
    graph = dataset[0]

    graph.numpy()
    node_idx = np.concatenate([graph.train_idx, graph.test_idx])
    node_idx, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, 2, graph.edge_index, graph.num_nodes, relabel_nodes=True)

    graph.num_nodes = node_idx.shape[0]
    graph.edge_index = edge_index
    graph.edge_type = graph.edge_type[edge_mask]
    graph.train_idx = mapping[:graph.train_idx.shape[0]]
    graph.test_idx = mapping[graph.train_idx.shape[0]:]
    graph.tensor()

    train_y = graph.train_y
    test_y = graph.test_y
    edge_index = graph.edge_index
    edge_type = graph.edge_type


    net = RGCN(feature_dim=graph.num_nodes,
               hidden_dim=args.hidden_dim,
               num_class=int(dataset.num_classes),
               num_relations=dataset.num_relations,
               name="RGCN")

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        'edge_index': edge_index,
        'edge_type': edge_type,
        'train_idx': graph.train_idx,
        'test_idx': graph.test_idx
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, train_y)
        val_acc = evaluate(net, data, test_y, metrics)

        print("Epoch [{:0>3d}] ".format(epoch + 1)\
              + "  train loss: {:.4f}".format(train_loss.item())\
              + "  val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

    net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict')
    test_acc = evaluate(net, data, test_y, metrics)
    print("Test acc:  {:.4f}".format(test_acc))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=50, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=16, help="dimention of hidden layers")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument('--num_bases', type=int, default=None, help='number of bases')
    parser.add_argument('--num_blocks', type=int, default=None, help='numbere of blocks')
    parser.add_argument("--aggregation", type=str, default='sum', help='aggregate type')
    parser.add_argument('--dataset', type=str, default='aifb', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    args = parser.parse_args()

    main(args)
