# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   gcn_trainer.py
@Time    :   2021/11/02 22:05:55
@Author  :   hanhui
"""

import os
os.environ['TL_BACKEND'] = 'tensorflow' # set your backend here, default `tensorflow`
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
# sys.path.insert(0, os.path.abspath('../../')) # adds path2gammagl to execute in command line.
import time
import tensorflow as tf
import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import GCNModel
from tensorlayerx.model import TrainOneStep, WithLoss

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self._backbone(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
        train_logits = logits[data['train_mask']]
        train_label = label[data['train_mask']]
        loss = self._loss_fn(train_logits, train_label)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self._backbone.trainable_weights]) * args.l2_coef # only support for tensorflow backend
        return loss + l2_loss

def evaluate(net, data, y, mask, metrics):
    net.set_eval()
    logits = net(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
    _logits = logits[mask]
    _label = y[mask]
    metrics.update(_logits, _label)
    acc = metrics.result()#[0]
    metrics.reset()
    return acc

def main(args):
    # load cora dataset
    if str.lower(args.dataset) not in ['cora','pubmed','citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    # dataset.process() # suggest to execute explicitly so far
    graph = dataset[0]
    graph.add_self_loop(n_loops=args.self_loops) # self-loop trick
    edge_weight = tlx.ops.convert_to_tensor(GCNModel.calc_gcn_norm(graph.edge_index, graph.num_nodes))
    x = graph.x
    edge_index = graph.edge_index
    y = tlx.argmax(graph.y,1)


    # build model
    best_model_path = os.path.abspath(args.best_model_path)
    if not os.path.exists(best_model_path):
        os.makedirs(os.path.abspath(best_model_path))

    net = GCNModel(feature_dim=x.shape[1],
                   hidden_dim=args.hidden_dim,
                   num_class=graph.y.shape[1],
                   keep_rate=args.keep_rate,
                   name="GCN")
    optimizer = tlx.optimizers.Adam(args.lr)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "x": x,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "train_mask": graph.train_mask,
        "test_mask": graph.test_mask,
        "val_mask": graph.val_mask,
        "num_nodes": graph.num_nodes,
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, y)
        val_acc = evaluate(net, data, y, data['val_mask'], metrics)

        print("Epoch [{:0>3d}] ".format(epoch+1)\
              + "  train loss: {:.4f}".format(train_loss)\
              + "  val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(best_model_path+net.name+".npz", format='npz_dict')

    net.load_weights(best_model_path+net.name+".npz", format='npz_dict')
    test_acc = evaluate(net, data, y, data['test_mask'], metrics)
    print("Test acc:  {:.4f}".format(test_acc))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=16, help="dimention of hidden layers")
    parser.add_argument("--keep_rate", type=float, default=0.5, help="keep_rate = 1 - drop_rate")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    args = parser.parse_args()

    # physical_gpus = tf.config.experimental.list_physical_devices('GPU')
    # if len(physical_gpus) > 0:
    #     # dynamic allocate gpu memory
    #     tf.config.experimental.set_memory_growth(physical_gpus[0], True)

    main(args)
