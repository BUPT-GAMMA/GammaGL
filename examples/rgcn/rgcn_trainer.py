# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/20 13:47
# @Author  : hanhui
# @FileName: trainer.py.py
import os

os.environ['TL_BACKEND'] = 'tensorflow'  # set your backend here, default `tensorflow`
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys

sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
from gammagl.models import RGCN
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.datasets import Entities, IMDB
import os.path as osp
import numpy as np
import gammagl.transforms as T
from gammagl.utils import mask_to_index


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self._backbone(data['edge_index'], data['edge_type'])
        train_logits = tlx.gather(logits, data['train_idx'])
        # train_y = y # tlx.gather(y, data['train_idx'])
        loss = self._loss_fn(train_logits, data['train_y'])
        return loss


def evaluate(net, data, y, metrics):
    net.set_eval()
    logits = net(data['edge_index'], data['edge_type'])
    _logits = tlx.gather(logits, data['test_idx'])
    _y = y  # tlx.gather(y, data['test_idx'])
    metrics.update(_logits, _y)
    acc = metrics.result()
    metrics.reset()
    return acc


def calculate_acc(logits, y, metrics):
    """
    Args:
        logits: node logits
        y: node labels
        metrics: tensorlayerx.metrics

    Returns:
        rst
    """

    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


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
        node_idx = -np.ones((num_nodes,))
        node_idx[subset] = np.arange(subset.shape[0])
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


def main(args):
    # load cora dataset
    if str.lower(args.dataset) not in ['aifb', 'mutag', 'bgs', 'am', 'imdb']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    if str.lower(args.dataset) == 'imdb':
        targetType = {
            'imdb': 'movie',
        }
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../IMDB')
        metapaths = [[('movie', 'actor'), ('actor', 'movie')],
                     [('movie', 'director'), ('director', 'movie')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edges=True,
                                   drop_unconnected_nodes=True)
        dataset = IMDB(path, transform=transform)
        heterograph = dataset[0]
        y = heterograph[targetType[str.lower(args.dataset)]].y
        num_classes = max(y) + 1

        edge_index_dict = {heterograph.edge_types[i]: heterograph.edge_stores[i]['edge_index'] for i in
                           range(len(heterograph.edge_stores))}

        train_idx = mask_to_index(heterograph['movie'].train_mask)
        test_idx = mask_to_index(heterograph['movie'].test_mask)
        val_idx = mask_to_index(heterograph['movie'].val_mask)

        train_y = tlx.gather(y, train_idx)
        test_y = tlx.gather(y, test_idx)
        val_y = tlx.gather(y, val_idx)

        edge_index = tlx.concat(
            [edge_index_dict[('movie', 'metapath_0', 'movie')], edge_index_dict[('movie', 'metapath_1', 'movie')]],
            axis=1)
        if tlx.BACKEND == 'tensorflow':
            edge_index = tlx.convert_to_tensor(edge_index, dtype=tlx.int32)
        else:
            edge_index = tlx.convert_to_tensor(edge_index, dtype=tlx.int64)
        edge_type0 = tlx.zeros(shape=(len(edge_index_dict[('movie', 'metapath_0', 'movie')][0]),), dtype=tlx.int64)
        edge_type1 = tlx.ones(shape=(len(edge_index_dict[('movie', 'metapath_1', 'movie')][0]),), dtype=tlx.int64)
        edge_type = tlx.concat([edge_type0, edge_type1], axis=0)

        data = {
            'edge_index': edge_index,
            'edge_type': edge_type,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'val_idx': val_idx,
            'train_y': train_y,
            'test_y': test_y,
            'val_y': val_y,
            'num_class': num_classes,
            'num_relations': 4,
            'num_nodes': tlx.convert_to_numpy(max(edge_index[0]) + 1).item()
        }

    else:
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

        data = {
            'edge_index': edge_index,
            'edge_type': edge_type,
            'train_idx': graph.train_idx,
            'test_idx': graph.test_idx,
            'train_y': graph.train_y,
            'test_y': graph.test_y,
            'num_class': int(dataset.num_classes),
            'num_relations': dataset.num_relations,
            'num_nodes': graph.num_nodes
        }

    net = RGCN(feature_dim=data['num_nodes'],
               hidden_dim=args.hidden_dim,
               num_class=data['num_class'],
               num_relations=data['num_relations'],
               name="RGCN")

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, data['train_y'])
        net.set_eval()
        logits = net(data['edge_index'], data['edge_type'])
        if str.lower(args.dataset) == 'imdb':
            val_logits = tlx.gather(logits, data['val_idx'])
            val_acc = calculate_acc(val_logits, data['val_y'], metrics)
        else:
            val_logits = tlx.gather(logits, data['test_idx'])
            val_acc = calculate_acc(val_logits, data['test_y'], metrics)
        # val_acc = evaluate(net, data, test_y, metrics)

        print("Epoch [{:0>3d}] ".format(epoch + 1) \
              + "  train loss: {:.4f}".format(train_loss.item()) \
              + "  val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

    net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict')
    if tlx.BACKEND == 'torch':
        net.to(data['edge_index'].device)
    net.set_eval()
    logits = net(data['edge_index'], data['edge_type'])
    test_logits = tlx.gather(logits, data['test_idx'])
    test_acc = calculate_acc(test_logits, data['test_y'], metrics)
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
