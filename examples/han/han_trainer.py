# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/16 25:16
# @Author  : clear
# @FileName: han_trainer.py
import os
os.environ['TL_BACKEND'] = 'tensorflow'  # set your backend here, default `tensorflow`
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
import os.path as osp
import gammagl.transforms as T
from gammagl.datasets import IMDB
from gammagl.models import HAN
import argparse
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss


def calc_loss(net, data, y, mask_name, loss_func=None, metrics=None):
    logits = net(data['x_dict'], data['edge_index_dict'], data['num_node_dict'])
    _logits = logits['movie'][data[mask_name]]
    _y = y[data[mask_name]] # tlx.gather(y, data['test_idx'])

    loss, acc = None, None
    if loss_func is not None:
        loss = loss_func(_logits, _y)
    if metrics is not None:
        metrics.update(_logits, _y)
        acc = metrics.result()
        metrics.reset()
    return loss, acc

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn, metrics):
        self.metrics = metrics
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        loss, self.train_acc = calc_loss(self._backbone, data, y, 'train_mask',
                            loss_func=self._loss_fn, metrics=self.metrics)
        return loss # only loss here

def main(args):
    # NOTE: ONLY IMDB DATASET
    # If you want to execute HAN on other dataset (e.g. ACM),
    # you will be needed to init `metepaths`
    # and set `movie` string with proper values.

    path = osp.join(osp.dirname(osp.realpath(__file__)), '../IMDB')
    metapaths = [[('movie', 'actor'), ('actor', 'movie')],
                 [('movie', 'director'), ('director', 'movie')]]
    transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edges=True,
                               drop_unconnected_nodes=True)
    dataset = IMDB(path, transform=transform)
    graph = dataset[0]
    y = graph['movie'].y


    net = HAN(
        in_channels=graph.x_dict['movie'].shape[1],
        out_channels=3, # graph.num_classes,
        metadata=graph.metadata(),
        hidden_channels=args.hidden_dim,
        heads=8
    )

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = tlx.losses.softmax_cross_entropy_with_logits
    semi_spvz_loss = SemiSpvzLoss(net, loss_func, metrics)
    train_one_step = TrainOneStep(semi_spvz_loss, optimizer, train_weights)

    # train test val = 400, 3478, 400
    data = {
        "x_dict": graph.x_dict,
        "edge_index_dict": graph.edge_index_dict,
        "train_mask": graph['movie'].train_mask,
        "test_mask": graph['movie'].test_mask,
        "val_mask": graph['movie'].val_mask,
        "num_node_dict": {'movie': graph['movie'].num_nodes},
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss, train_acc = train_one_step(data, y), semi_spvz_loss.train_acc
        net.set_eval()
        val_loss, val_acc = calc_loss(net, data, y,'val_mask',
                                      loss_func=loss_func, metrics=metrics)

        print("Epoch [{:0>3d}]  ".format(epoch + 1)
              + "   train_loss: {:.4f}".format(train_loss.item())
              + "   train_acc: {:.4f}".format(train_acc)
              + "   val_acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

    net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict')
    net.set_eval()
    test_loss, test_acc = calc_loss(net, data, y, 'test_mask',
                                    loss_func=loss_func, metrics=metrics)
    print("Test acc:  {:.4f}".format(test_acc))

if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.005, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=50, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=16, help="dimention of hidden layers")
    parser.add_argument("--l2_coef", type=float, default=1e-3, help="l2 loss coeficient")
    parser.add_argument("--heads", type=int, default=8, help="number of heads for stablization")
    parser.add_argument("--drop_rate", type=float, default=0.4, help="drop_rate")
    # parser.add_argument('--dataset', type=str, default='IMDB', help='dataset, not work')
    # parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset, not work")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    args = parser.parse_args()
    main(args)
