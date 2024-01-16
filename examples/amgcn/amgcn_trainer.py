# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   gcn_trainer.py
@Time    :   2021/11/02 22:05:55
@Author  :   hanhui
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import scipy.sparse as sp
import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import SFGCNModel
from gammagl.utils import add_self_loops, mask_to_index
from tensorlayerx.model import TrainOneStep, WithLoss
from sklearn.metrics.pairwise import cosine_similarity as cos

def knn(feat, num_node, k):
    adj = np.zeros((num_node, num_node), dtype=np.int64)
    dist = cos(tlx.to_device(feat, "CPU"))
    col = np.argpartition(dist, -(k + 1), axis=1)[:, -(k + 1):].flatten()
    adj[np.arange(num_node).repeat(k + 1), col] = 1
    adj = sp.coo_matrix(adj)
    return adj

def nll_loss_func(output, target):
    return -tlx.reduce_mean(tlx.gather(output, [range(tlx.get_tensor_shape(target)[0]), target]))

def common_loss(emb1, emb2):
    emb1 = emb1 - tlx.reduce_mean(emb1, axis=0, keepdims=True)
    emb2 = emb2 - tlx.reduce_mean(emb2, axis=0, keepdims=True)
    emb1 = tlx.l2_normalize(emb1, axis=1)
    emb2 = tlx.l2_normalize(emb2, axis=1)
    cov1 = tlx.matmul(emb1, tlx.transpose(emb1))
    cov2 = tlx.matmul(emb2, tlx.transpose(emb2))
    cost = tlx.reduce_mean((cov1 - cov2)**2)
    return cost

def loss_dependence(emb1, emb2, dim):
    R = tlx.eye(dim) - (1 / dim) * tlx.ones(shape=(dim, dim))
    K1 = tlx.matmul(emb1, tlx.transpose(emb1))
    K2 = tlx.matmul(emb2, tlx.transpose(emb2))
    RK1 = tlx.matmul(R, K1)
    RK2 = tlx.matmul(R, K2)
    HSIC = tlx.matmul(RK1, RK2)
    HSIC = tlx.reduce_sum(tlx.diag(HSIC))
    return HSIC

class SemiSpvzLoss(WithLoss):
    def __init__(self, net):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=None)

    def forward(self, data, y):
        logits, att, emb1, com1, com2, emb2, emb = self.backbone_network(data['x'], data['edge_index_s'], data['edge_index_f'])
        loss_class = nll_loss_func(tlx.gather(logits, data['train_idx']), tlx.gather(data['y'], data['train_idx']))
        loss_dep = (loss_dependence(emb1, com1, data['num_nodes']) + loss_dependence(emb2, com2, data['num_nodes'])) / 2
        loss_com = common_loss(com1, com2)
        loss = loss_class + data['beta'] * loss_dep + data['theta'] * loss_com

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
    # load datasets
    # set_device(5)
    if str.lower(args.dataset) not in ['cora','pubmed','citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    edge_index_f = knn(graph.x, graph.num_nodes, args.k)
    edge_index_f = tlx.convert_to_tensor([edge_index_f.row, edge_index_f.col], dtype=tlx.int64)
    edge_index_s, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes, n_loops=args.self_loops)
    
    # edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, graph.num_nodes))

    # for mindspore, it should be passed into node indices
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)

    net = SFGCNModel(num_feat=dataset.num_node_features,
                   num_class=dataset.num_classes,
                   num_hidden1=args.hidden1,
                   num_hidden2=args.hidden2,
                   dropout=args.drop_rate)

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index_s": edge_index_s,
        "edge_index_f": edge_index_f,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
        "beta": args.beta,
        "theta": args.theta
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, graph.y)
        net.set_eval()
        logits, att, emb1, com1, com2, emb2, emb = net(data['x'], data['edge_index_s'], data['edge_index_f'])
        val_logits = tlx.gather(logits, data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        val_acc = calculate_acc(val_logits, val_y, metrics)

        print("Epoch [{:0>3d}] ".format(epoch+1)\
              + "  train loss: {:.4f}".format(train_loss.item())\
              + "  val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path+net.name+".npz", format='npz_dict')

    net.load_weights(args.best_model_path+net.name+".npz", format='npz_dict')
    if tlx.BACKEND == 'torch':
        net.to(data['x'].device)
    net.set_eval()
    logits, att, emb1, com1, com2, emb2, emb = net(data['x'], data['edge_index_s'], data['edge_index_f'])
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)
    print("Test acc:  {:.4f}".format(test_acc))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--hidden1", type=int, default=32, help="dimention of hidden layers")
    parser.add_argument("--hidden2", type=int, default=16, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.5, help="drop_rate")
    parser.add_argument("--beta", type=float, default=0.000005, help="drop_rate")
    parser.add_argument("--theta", type=float, default=0.001, help="drop_rate")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    parser.add_argument("--k", type=int, default=7, help="dimention of hidden layers")
    # parser.add_argument("--n", type=int, default=10, help="dimention of hidden layers")
    parser.add_argument("--gpu", type = int, default=0)
    
    args = parser.parse_args()
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    main(args)
