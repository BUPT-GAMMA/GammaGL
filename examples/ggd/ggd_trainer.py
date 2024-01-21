import numpy as np
import scipy.sparse as sp

import os
import copy
import random
import argparse
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.datasets import Planetoid
from gammagl.datasets import Amazon
from gammagl.models import GGDModel
from gammagl.utils import add_self_loops, mask_to_index, calc_gcn_norm, to_scipy_sparse_matrix
from tensorlayerx.model import TrainOneStep, WithLoss

class GGDLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(GGDLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        aug_fts = aug_feature_dropout(data['features']) #augmentation on features
        idx = tlx.expand_dims(tlx.convert_to_tensor(np.random.permutation(data['num_nodes'])), 1)
        shuf_fts = tlx.zeros_like(aug_fts)
        shuf_fts = tlx.tensor_scatter_nd_update(shuf_fts, idx, aug_fts)
        lbl_1 = tlx.ones((data['batch_size'], data['num_nodes']))
        lbl_2 = tlx.zeros((data['batch_size'], data['num_nodes']))
        lbl = tlx.concat((lbl_1, lbl_2), 1)

        logits_1 = self.backbone_network(aug_fts, shuf_fts, data['edge_index'], data['edge_weight'], data['num_nodes'])
        loss_disc = self._loss_fn(tlx.sigmoid(logits_1), lbl)

        return loss_disc

class LogRegLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(LogRegLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['train_embs'])
        loss = self._loss_fn(logits, data['train_lbls'])
        return loss

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(in_features=ft_in, out_features=nb_classes, W_init=tlx.initializers.xavier_uniform())

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

def sparse_mx_to_edge_index(sparse_mx):
    """Convert a scipy sparse matrix to a tlx sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = tlx.convert_to_tensor(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = tlx.convert_to_tensor(sparse_mx.data)
    return indices, values

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = tlx.convert_to_numpy(tlx.reduce_sum(tlx.to_device(features, "cpu"), axis=1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(tlx.to_device(features, "cpu"))
    return tlx.convert_to_tensor(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def aug_random_edge(input_adj, drop_percent=0.1):
    drop_percent = drop_percent
    b = np.where(input_adj > 0,
                 np.random.choice(2, (input_adj.shape[0], input_adj.shape[0]), p=[drop_percent, 1 - drop_percent]),
                 input_adj)
    drop_num = len(input_adj.nonzero()[0]) - len(b.nonzero()[0])
    mask_p = drop_num / (input_adj.shape[0] * input_adj.shape[0] - len(b.nonzero()[0]))
    c = np.where(b == 0, np.random.choice(2, (input_adj.shape[0], input_adj.shape[0]), p=[1 - mask_p, mask_p]), b)

    return b

def aug_feature_dropout(input_feat, drop_percent=0.2):
    aug_input_feat = tlx.transpose(tlx.convert_to_tensor(copy.deepcopy(input_feat)))
    drop_feat_num = int(aug_input_feat.shape[0] * drop_percent)
    drop_idx = tlx.expand_dims(tlx.convert_to_tensor(random.sample([i for i in range(aug_input_feat.shape[0])], drop_feat_num)), 1)
    zeros = tlx.zeros((drop_feat_num, aug_input_feat.shape[1]))
    # pdb.set_trace()
    aug_input_feat = tlx.tensor_scatter_nd_update(aug_input_feat, drop_idx, zeros)
    return tlx.transpose(aug_input_feat)


if __name__ == '__main__':
    acc_results = []
    import warnings

    warnings.filterwarnings("ignore")

    #setting arguments
    parser = argparse.ArgumentParser('GGD')
    parser.add_argument('--classifier_epochs', type=int, default=100, help='classifier epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--np_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=500, help='Patience')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--l2_coef', type=float, default=0.0, help='l2 coef')
    parser.add_argument('--drop_prob', type=float, default=0.0, help='Tau value')
    parser.add_argument('--hid_units', type=int, default=512, help='Top-K value')
    parser.add_argument('--sparse', action='store_true', help='Whether to use sparse tensors')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name: cora, citeseer, pubmed, computer, photo')
    parser.add_argument('--num_hop', type=int, default=0, help='graph power')
    parser.add_argument('--n_trials', type=int, default=1, help='number of trails')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id, -1 means cpu")
    args = parser.parse_args()

    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    n_trails = args.n_trials
    acc_res = []
    for i in range(n_trails):

        dataset = args.dataset

        # training params
        batch_size = args.batch_size
        nb_epochs = args.np_epochs
        patience = args.patience
        classifier_epochs = args.classifier_epochs
        l2_coef = args.l2_coef
        drop_prob = args.drop_prob
        hid_units = args.hid_units
        num_hop = args.num_hop
        sparse = True
        nonlinearity = 'prelu'  # special name to separate parameters

        #load dataset
        if dataset in ['cora','citeseer','pubmed']:
            dataset = Planetoid(args.dataset_path, args.dataset)
            graph = dataset[0]
            edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes, n_loops=args.self_loops)
            features = graph.x
            labels = graph.y
            n_values = tlx.convert_to_numpy(tlx.reduce_max(labels) + 1).item()
            labels = tlx.gather(tlx.eye(n_values), labels)
            idx_train = mask_to_index(graph.train_mask)
            idx_test = mask_to_index(graph.test_mask)
            idx_val = mask_to_index(graph.val_mask)
            edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, graph.num_nodes))
            adj = to_scipy_sparse_matrix(edge_index, edge_weight)
        elif dataset in ['computers', 'photo']:
            dataset = Amazon(args.dataset_path, args.dataset)
            graph = dataset[0]
            edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes, n_loops=args.self_loops)
            features = tlx.convert_to_numpy(graph.x)
            labels = tlx.convert_to_numpy(graph.y)
            n_values = np.max(labels) + 1
            labels = np.eye(n_values)[labels]
            train_val_ratio = 0.2
            idx_train_val = random.sample(list(np.arange(features.shape[0])), int(train_val_ratio * features.shape[0]))
            remain_num = len(idx_train_val)
            idx_train = idx_train_val[remain_num//2:]
            idx_val = idx_train_val[:remain_num//2]
            idx_test = list(set(np.arange(features.shape[0])) - set(idx_train_val))
            mask = ['train_mask', 'test_mask', 'val_mask']
            for i, idx in enumerate([idx_train, idx_test, idx_val]):
                temp_mask = tlx.zeros((graph.num_nodes, ))
                temp_mask[idx] = 1
                graph[mask[i]] = temp_mask.bool()
            edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, graph.num_nodes))
            adj = to_scipy_sparse_matrix(edge_index, edge_weight)

        #preprocessing and initialisation
        features = preprocess_features(features)

        nb_nodes = tlx.get_tensor_shape(features)[0]
        nb_classes = tlx.get_tensor_shape(labels)[1]
        ft_size = tlx.get_tensor_shape(features)[1]

        original_features = tlx.expand_dims(features, 0)
        ggd = GGDModel(ft_size, hid_units, nb_classes)
        train_weights = ggd.trainable_weights
        optimiser_disc = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
        ggd_loss_func = GGDLoss(ggd, tlx.losses.binary_cross_entropy)
        train_one_step = TrainOneStep(ggd_loss_func, optimiser_disc, train_weights)

        data = {
            "features": features,
            "labels": labels,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "idx_train": idx_train,
            "idx_test": idx_test,
            "idx_val": idx_val,
            "num_nodes": graph.num_nodes,
            "batch_size": batch_size,
        }
        cnt_wait = 0
        best = 1e9
        best_t = 0
        features = tlx.expand_dims(features, 0)
        nb_feats = tlx.get_tensor_shape(features)[2]
        avg_time = 0
        counts = 0

        for epoch in range(nb_epochs):
            ggd.set_train()
            loss_disc = train_one_step(data, graph.y)
            print("Epoch ", epoch, ":\tloss: {:.4f}".format(loss_disc))

            if loss_disc < best:
                best = loss_disc
                best_t = epoch
                cnt_wait = 0
                ggd.save_weights('GGD.npz', format='npz_dict')
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                print('Early stopping!')
                break

        ggd.load_weights('GGD.npz', format='npz_dict')
        or_embeds, pr_embeds = ggd.embed(tlx.squeeze(original_features, axis=0), edge_index, edge_weight)
        embeds = or_embeds + pr_embeds
        train_embs = embeds[0, idx_train]
        val_embs = embeds[0, idx_val]
        test_embs = embeds[0, idx_test]

        labels = tlx.expand_dims(labels, axis=0)
        train_lbls = tlx.argmax(labels[0, idx_train], axis=1)
        val_lbls = tlx.argmax(labels[0, idx_val], axis=1)
        test_lbls = tlx.argmax(labels[0, idx_test], axis=1)

        data = {
            "train_embs": train_embs,
            "train_lbls": train_lbls,
        }

        tot = 0
        accs = []
        for _ in range(50):
            log = LogReg(tlx.get_tensor_shape(train_embs)[1], nb_classes)
            train_weights = log.trainable_weights
            log_loss_func = LogRegLoss(log, tlx.losses.softmax_cross_entropy_with_logits)
            opt = tlx.optimizers.Adam(lr=0.01, weight_decay=0.0)
            train_one_step = TrainOneStep(log_loss_func, opt, train_weights)

            pat_steps = 0
            best_acc = 0
            for _ in range(args.classifier_epochs):
                log.set_train()
                loss_disc = train_one_step(data, graph.y)

            log.set_eval()
            logits = log(test_embs)
            preds = tlx.argmax(logits, axis=1)
            acc = tlx.reduce_sum(preds == test_lbls).float() / tlx.get_tensor_shape(test_lbls)[0]
            accs.append(acc * 100)
            tot += acc

        accs = tlx.stack(accs)
        print(tlx.reduce_mean(accs))
        acc_results.append(tlx.convert_to_numpy(tlx.to_device(tlx.reduce_mean(accs), "cpu")))

    print("Test acc: ", np.mean(acc_results))

