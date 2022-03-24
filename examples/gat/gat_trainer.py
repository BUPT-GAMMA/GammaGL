# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   gat_trainer.py
@Time    :   2021/11/11 15:57:45
@Author  :   hanhui
"""

import os
os.environ['TL_BACKEND'] = 'tensorflow' # set your backend here, default `tensorflow`
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
# sys.path.insert(0, os.path.abspath('../../')) # adds path2gammagl to execute in command line.
import time
import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import GATModel
from gammagl.utils.config import Config
from tensorlayerx.model import TrainOneStep, WithLoss


def evaluate(net, data, node_label, mask, metrics):
    net.set_eval()
    logits = net(data['node_feat'], data['edge_index'], data['num_nodes'])
    _logits = logits[mask]
    _label = node_label[mask]
    metrics.update(_logits, _label)
    acc = metrics.result()  # [0]
    metrics.reset()
    return acc


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self._backbone(data['node_feat'], data['edge_index'], data['num_nodes'])
        train_logits = logits[data['train_mask']]
        train_label = label[data['train_mask']]
        loss = self._loss_fn(train_logits, train_label)
        return loss



def main(args):
    # load cora dataset
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid("../", args.dataset)
    dataset.process()  # suggest to execute explicitly so far
    graph = dataset[0]
    graph.add_self_loop(n_loops=1)  # self-loop trick
    edge_index = graph.edge_index
    node_feat = graph.x
    node_label = tlx.argmax(graph.y, 1)

    # configurate and build model
    cfg = Config(feature_dim=node_feat.shape[1],
                 hidden_dim=args.hidden_dim,
                 num_class=graph.y.shape[1],
                 heads=args.heads,
                 keep_rate=args.keep_rate)

    best_model_path = r'./best_models/'
    if not os.path.exists(best_model_path): os.makedirs(best_model_path)
    loss = tlx.losses.softmax_cross_entropy_with_logits
    optimizer = tlx.optimizers.Adam(args.lr)
    metrics = tlx.metrics.Accuracy()
    net = GATModel(cfg, name="GAT")
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, loss)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "node_feat": node_feat,
        "edge_index": edge_index,
        "train_mask": graph.train_mask,
        "test_mask": graph.test_mask,
        "val_mask": graph.val_mask,
        "num_nodes": graph.num_nodes,
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, node_label)
        val_acc = evaluate(net, data, node_label, data['val_mask'], metrics)

        print("Epoch [{:0>3d}]  ".format(epoch + 1)\
              + "   train loss: {:.4f}".format(train_loss)\
              + "   val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(best_model_path+net.name+".npz", format='npz_dict')

    net.load_weights(best_model_path+net.name+".npz", format='npz_dict')
    test_acc = evaluate(net, data, node_label, data['test_mask'], metrics)
    print("Test acc:  {:.4f}".format(test_acc))

if __name__ == "__main__":
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.005, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=8, help="dimention of hidden layers")
    parser.add_argument("--keep_rate", type=float, default=0.4, help="keep_rate = 1 - drop_rate")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument("--heads", type=int, default=8, help="number of heads for stablization")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument("--self_loop", action='store_false', default=True, help="graph self-loop (default=True)")
    args = parser.parse_args()

    # physical_gpus = tf.config.experimental.list_physical_devices('GPU')
    # if len(physical_gpus) > 0:
    #     # dynamic allocate gpu memory
    #     tf.config.experimental.set_memory_growth(physical_gpus[0], True)

    main(args)
