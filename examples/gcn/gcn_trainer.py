# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   gcn_trainer.py
@Time    :   2021/11/02 22:05:55
@Author  :   hanhui
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['CUDA_VISIBLE_DEVICES']='7'
# os.environ['TL_BACKEND'] = 'tensorflow'
os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'torch'
# import tensorflow as tf

import sys
sys.path.insert(0, os.path.abspath('../../')) # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import GCNModel
from gammagl.utils.loop import add_self_loops
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.utils.norm import calc_gcn_norm
# from pyinstrument import Profiler

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        loss = clac_acc_loss(self.backbone_network, data, 'train_idx', loss_func=self._loss_fn)
        return loss


def clac_acc_loss(net, data, idx_type, loss_func=None, metrics=None):
    logits = net(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
    _logits = tlx.gather(logits, data[idx_type])
    _y = tlx.gather(data['y'], data[idx_type])

    if loss_func is not None:
        loss = loss_func(_logits, _y)
        return loss

    if metrics is not None:
        metrics.update(_logits, _y)
        acc = metrics.result()
        metrics.reset()
        return acc


def main(args):
    # load datasets
    if str.lower(args.dataset) not in ['cora','pubmed','citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes, n_loops=args.self_loops)
    edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, graph.num_nodes))
    x = graph.x
    y = tlx.argmax(graph.y, axis=1)
    graph.train_idx = tlx.convert_to_tensor([i for i, v in enumerate(graph.train_mask) if v], dtype=tlx.int64)
    graph.test_idx = tlx.convert_to_tensor([i for i, v in enumerate(graph.test_mask) if v], dtype=tlx.int64)
    graph.val_idx = tlx.convert_to_tensor([i for i, v in enumerate(graph.val_mask) if v], dtype=tlx.int64)

    # pf = Profiler()
    # pf.start()
    net = GCNModel(feature_dim=x.shape[1],
                   hidden_dim=args.hidden_dim,
                   num_class=dataset.num_classes,
                   drop_rate=args.drop_rate,
                   name="GCN")

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "x": x,
        'y': y,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "train_mask": graph.train_mask,
        "test_mask": graph.test_mask,
        "val_mask": graph.val_mask,
        "train_idx": graph.train_idx,
        "test_idx": graph.test_idx,
        "val_idx": graph.val_idx,
        "num_nodes": graph.num_nodes,
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, y)
        net.set_eval()
        val_acc = clac_acc_loss(net, data, 'val_idx', metrics=metrics)

        print("Epoch [{:0>3d}] ".format(epoch+1)\
              + "  train loss: {:.4f}".format(train_loss.item())\
              + "  val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path+net.name+"_"+args.dataset+".npz", format='npz_dict')

    net.load_weights(args.best_model_path+net.name+"_"+args.dataset+".npz", format='npz_dict')
    if os.environ['TL_BACKEND'] == 'torch':
        net.to(data['x'].device)
    test_acc = clac_acc_loss(net, data, 'test_idx', metrics=metrics)
    print("Test acc:  {:.4f}".format(test_acc))
    # pf.stop()
    # print(pf.output_text(unicode=True, color=True))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=16, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.5, help="drop_rate")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    args = parser.parse_args()

    main(args)
