# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
import sys
sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
import os.path as osp
import gammagl.transforms as T
from gammagl.datasets import IMDB
from gammagl.models import HPN
from gammagl.utils import mask_to_index, set_device
from tensorlayerx.model import TrainOneStep, WithLoss

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x_dict'], data['edge_index_dict'], data['num_nodes_dict'])
        train_logits = tlx.gather(logits['movie'], data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss

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
    # NOTE: ONLY IMDB DATASET
    # If you want to execute HPN on other dataset (e.g. ACM),
    # you will be needed to init `metepaths`
    # and set `movie` string with proper values.

    # set_device(args.gpu)
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '../IMDB')
    metapaths = [[('movie', 'actor'), ('actor', 'movie')],
                 [('movie', 'director'), ('director', 'movie')]]
    transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edges=True,
                               drop_unconnected_nodes=True)
    dataset = IMDB(args.dataset_path, transform=transform)
    graph = dataset[0]
    y = graph['movie'].y

    # for mindspore, it should be passed into node indices
    train_idx = mask_to_index(graph['movie'].train_mask,)
    test_idx = mask_to_index(graph['movie'].test_mask)
    val_idx = mask_to_index(graph['movie'].val_mask)

    net = HPN(
        in_channels=graph.x_dict['movie'].shape[1],
        out_channels=3, # graph.num_classes,
        metadata=graph.metadata(),
        drop_rate=args.drop_rate,
        hidden_channels=args.hidden_dim,
        iter_K=args.iter_K,
        alpha=args.alpha,
        name = 'hpn',
    )

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = tlx.losses.softmax_cross_entropy_with_logits
    semi_spvz_loss = SemiSpvzLoss(net, loss_func)
    train_one_step = TrainOneStep(semi_spvz_loss, optimizer, train_weights)

    # train test val = 400, 3478, 400
    data = {
        "x_dict": graph.x_dict,
        "y":y,
        "edge_index_dict": graph.edge_index_dict,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes_dict": {'movie': graph['movie'].num_nodes},
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, y)
        net.set_eval()
        logits = net(data['x_dict'], data['edge_index_dict'], data['num_nodes_dict'])
        val_logits = tlx.gather(logits['movie'], data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        val_acc = calculate_acc(val_logits, val_y, metrics)

        print("Epoch [{:0>3d}]  ".format(epoch + 1)
              + "   train_loss: {:.4f}".format(train_loss.item())
              # + "   train_acc: {:.4f}".format(train_acc)
              + "   val_acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

    net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict')
    net.set_eval()
    logits = net(data['x_dict'], data['edge_index_dict'], data['num_nodes_dict'])
    test_logits = tlx.gather(logits['movie'], data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)
    print("Test acc:  {:.4f}".format(test_acc))
    with open("output.txt", "a") as f:
        f.write(tlx.BACKEND + ":\n")
        f.write("lr: " + str(args.lr) + ", " + \
            "hidden_dim: " + str(args.hidden_dim) + ", " + \
            "l2_coef: " + str(args.l2_coef) + "\n")
        f.write("iter_K: " + str(args.iter_K) + ", " + \
            "drop_rate: " + str(args.drop_rate) + ", " + \
            "alpha: " + str(args.alpha) + "\n")
        f.write("Test acc:  {:.4f}\n\n".format(test_acc))

if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=512, help="dimention of hidden layers")
    parser.add_argument("--l2_coef", type=float, default=1e-3, help="l2 loss coeficient")
    parser.add_argument("--iter_K", type=int, default=1, help="number K of iteration")
    parser.add_argument("--drop_rate", type=float, default=0.4, help="drop_rate")
    parser.add_argument("--alpha", type=float, default=0.3, help="alpha")
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    # parser.add_argument('--dataset', type=str, default='IMDB', help='dataset')
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    args = parser.parse_args()
    main(args)
