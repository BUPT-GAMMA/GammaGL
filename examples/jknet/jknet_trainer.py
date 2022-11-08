# -*- coding: utf-8 -*-
"""
@File   : jknet_trainer.py
@Time   : 2022/4/10 11:16 A.M.
@Author : Jia Yiming
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ['TL_BACKEND'] = 'torch'


import sys
sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import JKNet
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.utils import add_self_loops, calc_gcn_norm, mask_to_index, set_device


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
        train_logits = tlx.gather(logits, data['train_idx'])
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
    # load cora dataset
    # set_device(5)
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    dataset.process()  # suggest to execute explicitly so far
    graph = dataset[0]

    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes, n_loops=args.self_loops)
    edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, graph.num_nodes))

    useful_node = 0
    useful_index = []
    useful_mask = []
    for i in range(graph.num_nodes):
        new = graph.train_mask[i] or graph.test_mask[i] or graph.val_mask[i]
        useful_mask.append(new)
        if new:
            useful_index.append(i)
            useful_node += 1
    useful_mask = tlx.convert_to_tensor(useful_mask)
    train_num = int(useful_node * 0.6)
    val_num = int(useful_node * 0.2)
    test_num = useful_node - train_num - val_num

    graph.train_mask = graph.train_mask.cpu().numpy()
    graph.val_mask = graph.val_mask.cpu().numpy()
    graph.test_mask = graph.test_mask.cpu().numpy()


    graph.train_mask[:] = False
    graph.train_mask[useful_index[:train_num]] = True
    graph.val_mask[:] = False
    graph.val_mask[useful_index[train_num:train_num + val_num]] = True
    graph.test_mask[:] = False
    graph.test_mask[useful_index[train_num + val_num:train_num + val_num + test_num]] = True

    graph.tensor()

    # for mindspore, it should be passed into node indices
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)

    net = JKNet(dataset=dataset,
                mode=args.mode,
                num_layers=args.iter_K,
                drop=args.drop_rate)

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, graph.y)

        net.set_eval()
        logits = net(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
        val_logits = tlx.gather(logits, data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        val_acc = calculate_acc(val_logits, val_y, metrics)

        print("Epoch [{:0>3d}] ".format(epoch + 1) \
              + "  train loss: {:.4f}".format(train_loss.item()) \
              + "  val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

    net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict')
    if tlx.BACKEND == 'torch':
        net.to(data['x'].device)

    net.set_eval()
    logits = net(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)
    print("Test acc:  {:.4f}".format(test_acc))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learnin rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Adam weight decay")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=16, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.5, help="drop rate")
    parser.add_argument("--iter_K", type=int, default=6, help="number K of iteration")
    parser.add_argument("--l2_coef", type=float, default=1e-3, help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    parser.add_argument("--mode", type=str, default='max', help="mode of jumping knowledge, optional=['max', 'cat', 'lstm']")

    args = parser.parse_args()
    main(args)