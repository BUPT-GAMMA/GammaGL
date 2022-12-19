import os
#os.environ['CUDA_VISIBLE_DEVICES']='1'
#os.environ['TL_BACKEND'] = 'paddle'
#os.environ['TL_BACKEND'] = 'torch'

import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.abspath('../../')) # adds path2gammagl to execute in command line.

import argparse
import tensorlayerx as tlx
from sklearn.metrics import roc_auc_score, average_precision_score
from gammagl.datasets import Planetoid
from gammagl.models import VGAEModel, GAEModel
from gammagl.transforms import mask_test_edges, sparse_to_tuple
from gammagl.utils import add_self_loops, calc_gcn_norm, mask_to_index, set_device
from tensorlayerx.model import TrainOneStep, WithLoss


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        preds, mu, logstd = self.backbone_network(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
        #cost = data['norm'] * self._loss_fn(tlx.sigmoid(preds), data['labels'])  #test
        loss = self._loss_fn(data, preds, mu, logstd)
        return loss


def weighted_cross_entropy_with_logits(logits, labels, pos_weight):

    #labels * -log(sigmoid(logits)) * pos_weight + (1 - labels) * -log(1 - sigmoid(logits))
    log_weight = 1 + (pos_weight - 1) * labels
    return tlx.add((1 - labels) * logits,
        log_weight * (tlx.log(tlx.exp(-tlx.abs(logits)) + 1) +
                      tlx.relu(-logits,)))


def get_loss_vgae(data, preds, mu, logstd):

    #cost = data['norm'] * tlx.losses.sigmoid_cross_entropy(preds, data['labels'])
    cost = data['norm'] * tlx.reduce_mean(weighted_cross_entropy_with_logits(preds, data['labels'], pos_weight=data['pos_weight']))
    KLD = -0.5 / data['num_nodes'] * tlx.reduce_mean(tlx.reduce_sum(1 + 2 * logstd - tlx.pow(mu, 2) - tlx.pow(tlx.exp(logstd), 2), 1))
    return cost + KLD


def get_loss_gae(data, preds, mu, logstd):

    #cost = data['norm'] * tlx.losses.sigmoid_cross_entropy(preds, data['labels'])
    cost = data['norm'] * tlx.reduce_mean(weighted_cross_entropy_with_logits(preds, data['labels'], pos_weight=data['pos_weight']))
    return cost


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


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

    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]

    row = np.array(graph.edge_index[0])
    col = np.array(graph.edge_index[1])
    data1 = np.ones((graph.edge_index.shape[1], ))
    adj = sp.csr_matrix((data1, (row, col)), shape=(graph.num_nodes, graph.num_nodes), dtype=np.int32)  #test

    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    features = graph.x
    if args.features == 0:  #no feartures
        features = tlx.eye(graph.num_nodes, graph.num_nodes)  # featureless

    # Some preprocessing
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = tlx.convert_to_tensor(adj_label.toarray(), tlx.float32) #test

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    pos_weight_tensor = tlx.convert_to_tensor(pos_weight * np.ones((1, adj.shape[0])), tlx.float32)

    coords = sparse_to_tuple(adj)[0].T
    coords = tlx.convert_to_tensor(coords, tlx.int64)

    edge_index, _ = add_self_loops(coords, num_nodes=graph.num_nodes, n_loops=args.self_loops)
    edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, graph.num_nodes))

    # for mindspore, it should be passed into node indices
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)

    if args.model == 'VGAE':
        net = VGAEModel(feature_dim=features.shape[1], hidden1_dim=args.hidden1_dim, hidden2_dim=args.hidden2_dim,
                        drop_rate=args.drop_rate, num_layers=args.num_layers, norm=args.norm, name="VGAE")
    else:
        net = GAEModel(feature_dim=features.shape[1], hidden1_dim=args.hidden1_dim, hidden2_dim=args.hidden2_dim,
                        drop_rate=args.drop_rate, num_layers=args.num_layers, norm=args.norm, name="GAE")


    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    #metrics = tlx.metrics.Auc()
    train_weights = net.trainable_weights

    if args.model == 'VGAE':
        loss = get_loss_vgae
    else:
        loss = get_loss_gae
    loss_func = SemiSpvzLoss(net, loss)
    #loss_func = SemiSpvzLoss(model, tlx.losses.binary_cross_entropy)  #test
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "x": features,
        "y": graph.y,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
        "labels": adj_label,
        "pos_weight": pos_weight_tensor,
        "norm": norm
    }

    best_val_ap = 0
    hidden_emb = None
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, graph.y)
        net.set_eval()

        mu = net(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])[1]
        hidden_emb = tlx.convert_to_numpy(mu)
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        print("Epoch [{:0>3d}] ".format(epoch + 1) \
              + "  train loss: {:.4f}".format(train_loss.item()) \
              + "  val ap: {:.4f}".format(ap_curr))

        # save best model on evaluation set
        if ap_curr > best_val_ap:
            best_val_ap = ap_curr
            net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

    net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict')
    if tlx.BACKEND == 'torch':
        net.to(data['x'].device)
    net.set_eval()

    mu = net(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])[1]

    hidden_emb = tlx.convert_to_numpy(mu)
    roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)

    print("Test AUC score:  {:.4f}".format(roc_curr))
    print("Test AP score:  {:.4f}".format(ap_curr))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--hidden1_dim", type=int, default=32, help="dimention of hidden layers1")
    parser.add_argument("--hidden2_dim", type=int, default=16, help="dimention of hidden layers2")
    parser.add_argument("--drop_rate", type=float, default=0., help="drop_rate")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--norm", type=str, default='none', help="how to apply the normalizer.")
    parser.add_argument("--l2_coef", type=float, default=0., help="l2 loss coeficient")  #5e-4
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    parser.add_argument("--features", type=int, default=1, help="Whether to use features (1) or not (0)")
    parser.add_argument("--model", type=str, default='VGAE', help="Model string")
    args = parser.parse_args()

    main(args)
