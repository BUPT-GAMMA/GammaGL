# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/11/8 23:47
# @Author  : yijian
# @FileName: compgcn_trainer.py
import os
# os.environ['TL_BACKEND'] = 'mindspore'  # set your backend here, default `torch`
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
from gammagl.models import CompGCN
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.datasets import Entities
import os.path as osp
import numpy as np
# tlx.set_device("GPU", 1)


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
    _y = y # tlx.gather(y, data['test_idx'])
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

def main(args):
    # load cora dataset
    if str.lower(args.dataset) not in ['aifb', 'mutag', 'bgs', 'am']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../Entities')
    dataset = Entities(path, args.dataset)
    graph = dataset[0]

    train_y = graph.train_y
    test_y = graph.test_y
    edge_index = graph.edge_index
    edge_type = graph.edge_type


    net = CompGCN(feature_dim=32,
               hidden_dim=args.hidden_dim,
               num_class=int(dataset.num_classes),
               num_relations=dataset.num_relations,
               num_entity = dataset[0].num_nodes,
               op = args.op,
               name="CompGCN")

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    #Preprocess Entity Index and Relation Index,
    #the first half of edge_index and edge_type are output
    #the second half of edge_index and edge_type are inverse
    #for edge type of initial dataset,even index is out edge, odd index is inverse edge
    #we must preprocess edge type
    edge_type = tlx.ops.convert_to_numpy(edge_type)
    edge_index = tlx.ops.convert_to_numpy(edge_index)
    edge_in_index = [[], []]
    edge_out_index = [[], []]
    edge_in_type = []
    edge_out_type = []
    for i in range(0, edge_type.shape[0]):
        if edge_type[i] % 2 == 0:
            edge_out_index[0].append(edge_index[0][i])
            edge_out_index[1].append(edge_index[1][i])
            edge_out_type.append(edge_type[i])
        else:
            edge_in_index[0].append(edge_index[0][i])
            edge_in_index[1].append(edge_index[1][i])
            edge_in_type.append(edge_type[i])

    edge_index = [[], []]

    edge_index[0] = edge_in_index[0] + edge_out_index[0]
    edge_index[1] = edge_in_index[1] + edge_out_index[1]
    edge_type = edge_in_type + edge_out_type

    edge_index = tlx.ops.convert_to_tensor(edge_index)
    edge_type = tlx.ops.convert_to_tensor(edge_type, dtype = tlx.int64)


    data = {
        'edge_index': edge_index,
        'edge_type': edge_type,
        'train_idx': graph.train_idx,
        'test_idx': graph.test_idx,
        'train_y': graph.train_y,
        'test_y': graph.test_y,
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, train_y)
        net.set_eval()
        logits = net(data['edge_index'], data['edge_type'])
        val_logits = tlx.gather(logits, data['test_idx'])
        val_acc = calculate_acc(val_logits, data['test_y'], metrics)
        # val_acc = evaluate(net, data, test_y, metrics)

        print("Epoch [{:0>3d}] ".format(epoch + 1)\
              + "  train loss: {:.4f}".format(train_loss.item())\
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
    parser.add_argument("--lr", type=float, default=0.015, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=500, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=16, help="dimention of hidden layers")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument('--num_bases', type=int, default=None, help='number of bases')
    parser.add_argument('--num_blocks', type=int, default=None, help='numbere of blocks')
    parser.add_argument("--aggregation", type=str, default='mean', help='aggregate type')
    parser.add_argument('--dataset', type=str, default='aifb', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--op", type=str, default='sub', help="op between entity and relation,sub or mult")
    args = parser.parse_args()

    main(args)
