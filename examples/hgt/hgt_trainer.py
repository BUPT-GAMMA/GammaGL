# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   hgt_trainer.py
@Time    :   2022/7/1 8:05:55
@Author  :   Zhang Zhongjian
"""

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# os.environ['TL_BACKEND'] = 'tensorflow'  # set your backend here, default `tensorflow`
import sys

sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
import tensorlayerx as tlx
import os.path as osp
import argparse
from gammagl.datasets import HGBDataset, IMDB
from gammagl.models import HGTModel
import gammagl.transforms as T
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.utils import mask_to_index

if tlx.BACKEND == 'torch':  # when the backend is torch and you want to use GPU
    try:
        tlx.set_device(device='GPU', id=1)
    except:
        print("GPU is not available")


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


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x_dict'], data['edge_index_dict'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss


def main(args):
    if (str.lower(args.dataset) not in ['imdb', 'dblp']):
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
    else:
        targetType = {
            'dblp': 'author',
        }
        dataset = HGBDataset(args.dataset_path, args.dataset)
        heterograph = dataset[0]
        y = heterograph[targetType[str.lower(args.dataset)]].y
        num_classes = (max(y) - min(y)) + 1
    edge_index_dict = {heterograph.edge_types[i]: heterograph.edge_stores[i]['edge_index'] for i in
                       range(len(heterograph.edge_stores))}
    x_dict = {node_type: heterograph[node_type].x for node_type in heterograph.node_types}
    val_ratio = 0.2
    train = mask_to_index(heterograph[targetType[str.lower(args.dataset)]].train_mask)
    split = int(train.shape[0] * val_ratio)
    train_idx = train[split:]
    val_idx = train[:split]
    test_idx = mask_to_index(heterograph[targetType[str.lower(args.dataset)]].test_mask)
    data = {
        'x_dict': x_dict,
        'y': y,
        'edge_index_dict': edge_index_dict,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }

    net = HGTModel(data=heterograph, hidden_channels=args.hidden_dim, out_channels=num_classes,
                   num_heads=args.heads, num_layers=args.num_layers,
                   target_node_type=targetType[str.lower(args.dataset)], drop_rate=args.drop_rate)

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights
    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, data['y'])
        net.set_eval()
        logits = net(data['x_dict'], data['edge_index_dict'])
        val_logits = tlx.gather(logits, data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        val_acc = calculate_acc(val_logits, val_y, metrics)
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        train_acc = calculate_acc(train_logits, train_y, metrics)

        test_logits = tlx.gather(logits, data['test_idx'])
        test_y = tlx.gather(data['y'], data['test_idx'])
        test_acc = calculate_acc(test_logits, test_y, metrics)
        print("Epoch [{:0>3d}] ".format(epoch + 1) \
              + "  train loss: {:.4f}".format(train_loss.item()) \
              + "  train acc: {:.4f}".format(train_acc) \
              + "  val acc: {:.4f}".format(val_acc) \
              + "  Test acc:  {:.4f}".format(test_acc))
        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path + "HGT_" + args.dataset + ".npz", format='npz_dict')

    net.load_weights(args.best_model_path + "HGT_" + args.dataset + ".npz", format='npz_dict')
    net.set_eval()
    logits = net(data['x_dict'], data['edge_index_dict'])
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)
    print("Test acc:  {:.4f}".format(test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="dimention of hidden layers")
    parser.add_argument("--l2_coef", type=float, default=1e-6, help="l2 loss coeficient")
    parser.add_argument("--heads", type=int, default=4, help="number of heads for stablization")
    parser.add_argument("--num_layers", type=int, default=2, help="number of hgt layers")
    parser.add_argument("--drop_rate", type=float, default=0.5, help="drop_rate")
    parser.add_argument('--dataset', type=str, default='IMDB', help='dataset, not work')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset, not work")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    args = parser.parse_args()
    main(args)
