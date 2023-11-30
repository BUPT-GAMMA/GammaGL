"""
@File    :   spec_trainer.py
@Time    :   2023/10/15
@Author  :   ZZH
"""

import os
# os.environ['TL_BACKEND'] = "torch"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorlayerx as tlx
import argparse
import gammagl.transforms as T
import numpy as np
from numpy.linalg import eigh
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.models.specformer import Specformer
from gammagl.datasets import WikipediaNetwork, Planetoid
try:
    import cPickle as pickle
except ImportError:
    import pickle

def feature_normalize(x):
    x = np.array(x)
    rowsum = x.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, 1, 1e10)
    return x / rowsum

def normalize_graph(g):
    r"""
    Parameters
    ----------
    g: Tensor
        The input adjacent matrix.

    Returns
    -------
    Tensor
        The Laplacian matrix
    """
    g = np.array(g)
    g = g + g.T
    g[g > 0.] = 1.0
    deg = g.sum(axis=1).reshape(-1)
    deg[deg == 0.] = 1.0
    deg = np.diag(deg ** -0.5)
    adj = np.dot(np.dot(deg, g), deg)
    L = np.eye(g.shape[0]) - adj
    return L

def eigen_decompositon(g):
    r"""
    Parameters
    ----------
    g: Tensor
        The input adjacent matrix.

    Returns
    -------
    numpy.array, numpy.array
        The eigenvalue and eigenvectors of the Laplacian matrix.
    """
    g = normalize_graph(g)
    e, u = eigh(g)
    return e, u

def get_split(y, nclass, seed=0):

    percls_trn = int(round(0.6 * len(y) / nclass))
    val_lb = int(round(0.2 * len(y)))

    indices = []
    for i in range(nclass):
        h = tlx.convert_to_numpy((y == i))
        res = np.nonzero(h)
        res = np.array(res).reshape(-1)
        index = tlx.convert_to_tensor(res)
        n = tlx.get_tensor_shape(index)[0]
        index = tlx.gather(index, np.random.permutation(n))
        indices.append(index)

    train_index = tlx.concat(values=[i[:percls_trn] for i in indices], axis=0)
    rest_index = tlx.concat(values=[i[percls_trn:] for i in indices], axis=0)
    m = tlx.get_tensor_shape(rest_index)[0]
    index2 = tlx.convert_to_tensor(np.random.permutation(m))
    index2 = tlx.cast(index2, dtype=tlx.int64)

    rest_index = tlx.gather(rest_index, index2)
    valid_index = rest_index[:val_lb]
    test_index = rest_index[val_lb:]

    return train_index, valid_index, test_index

def calculate_acc(logits, y, metrics):
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, e, u):
        logits = self.backbone_network(data['x'], data['edge_index'], e, u)

        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss

def main(args):
    # 3 datasets: Chameleon, Squirrel, Cora
    if args.dataset == "cora":
        dataset = Planetoid(root="./", name=args.dataset, split="full")
    else:# Chameleon, Squirrel
        dataset = WikipediaNetwork(root="./", name=args.dataset, transform=T.NormalizeFeatures())

    graph = dataset[0]
    edge_index = graph.edge_index
    train_idx, val_idx, test_idx = get_split(y=graph.y, nclass=dataset.num_classes)

    net = Specformer(nclass=dataset.num_classes, n_feat=graph.num_node_features, n_layer=args.n_layer,
                     hidden_dim=args.hidden_dim, n_heads=args.n_heads,
                     tran_dropout=args.tran_dropout, feat_dropout=args.feat_dropout, prop_dropout=args.prop_dropout)
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights
    loss_func = SemiSpvzLoss(net=net,
                             loss_fn=tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)
    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": edge_index,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
    }
    best_val_acc = 0.0

    num_nodes = graph.num_nodes
    adj = np.zeros((num_nodes, num_nodes))
    source = edge_index[0]
    target = edge_index[1]
    for i in range(len(source)):
        adj[source[i], target[i]] = 1.
        adj[target[i], source[i]] = 1.

    # adj is the adjacent matrix of graph
    e, u = eigen_decompositon(adj)
    e = tlx.convert_to_tensor(e)
    u = tlx.convert_to_tensor(u)

    for now_epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, e, u)
        net.set_eval()
        logits = net(data['x'], data['edge_index'], e, u)
        val_logits = tlx.gather(logits, data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        val_acc = calculate_acc(val_logits, val_y, metrics)
        #every epoch print val acc
        print("Epoch [{:0>3d}] ".format(now_epoch + 1) \
              + "  train loss: {:.4f}".format(train_loss.item()) \
              + "  val acc: {:.4f}".format(val_acc))

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights("./"+ net.name +"__" +args.dataset +".npz", format='npz_dict')


    # get the best model parameter
    net.load_weights("./"+ net.name +"__" +args.dataset +".npz", format='npz_dict')
    net.set_eval()
    logits = net(data['x'], data['edge_index'], e, u)
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)
    print("Test acc: {:.4f}".format(test_acc))
    return test_acc


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight_decay")
    parser.add_argument("--n_epoch", type=int, default=500, help="number of epoch")
    parser.add_argument("--n_layer", type=int, default=2, help="number of Speclayers")
    parser.add_argument("--n_heads", type=int, default=4, help="number of heads")
    parser.add_argument("--hidden_dim", type=int, default=32, help="dimension of hidden layers")
    parser.add_argument('--dataset', type=str, default='chameleon', help='dataset')
    parser.add_argument("--tran_dropout", type=float, default=0.2, help="tran_dropout")
    parser.add_argument("--feat_dropout", type=float, default=0.6, help="feat_dropout")
    parser.add_argument("--prop_dropout", type=float, default=0.2, help="prop_dropout")

    args = parser.parse_args()

    main(args)
