import os

os.environ['TL_BACKEND'] = 'torch'
import argparse

import torch
from aug import random_aug
import numpy as np
import torch as th
import tensorlayerx as tlx
import warnings
from gammagl.data import Graph
from gammagl.utils import add_self_loops

from sklearn.metrics import f1_score
import scipy.sparse as sp
from params import set_params
from model import DGIModel, LogReg

warnings.filterwarnings('ignore')

args = set_params('cora')  # cora/citeseer/pubmed/flickr/blog

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'
'''
## random seed ##
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
'''

own_str = args.dataname
print(own_str)


def coo_to_edge_index(coo_m):
    adj = torch.tensor(coo_m.todense())
    adj = sp.coo_matrix(adj)
    indices = np.vstack((adj.row, adj.col))
    edge_index = torch.LongTensor(indices)
    return edge_index


def sinkhorn(K, dist, sin_iter):
    # make the matrix sum to 1
    u = np.ones([len(dist), 1]) / len(dist)  # dist 行列的数值分布
    K_ = sp.diags(1. / dist) * K
    dist = dist.reshape(-1, 1)

    for it in range(sin_iter):
        u = 1. / K_.dot(dist / (K.T.dot(u)))
    v = dist / (K.T.dot(u))
    delta = np.diag(u.reshape(-1)).dot(K).dot(np.diag(v.reshape(-1)))
    return delta


def plug(theta, laplace, delta_add, delta_dele, epsilon, dist, sin_iter, c_flag=False):
    C = (1 - theta) * laplace.A
    if c_flag:
        C = laplace.A
    K_add = np.exp(2 * (C * delta_add).sum() * C / epsilon)
    K_dele = np.exp(-2 * (C * delta_dele).sum() * C / epsilon)

    delta_add = sinkhorn(K_add, dist, sin_iter)

    delta_dele = sinkhorn(K_dele, dist, sin_iter)
    return delta_add, delta_dele


def update(theta, epoch, total):
    theta = theta - theta * (epoch / total)
    return theta


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(np.abs(adj.A).sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)

def get_dataset(path, dataname, scope_flag):
    adj = sp.load_npz(path + "/adj.npz")

    feat = sp.load_npz(path + "/feat.npz").A
    if dataname != 'blog':
        feat = torch.Tensor(preprocess_features(feat))
    else:
        feat = torch.Tensor(feat)
    num_features = feat.shape[-1]
    label = torch.LongTensor(np.load(path + "/label.npy"))
    idx_train20 = np.load(path + "/train20.npy")
    idx_train10 = np.load(path + "/train10.npy")
    idx_train5 = np.load(path + "/train5.npy")
    idx_train = [idx_train5, idx_train10, idx_train20]
    idx_val = np.load(path + "/val.npy")
    idx_test = np.load(path + "/test.npy")
    num_class = label.max() + 1

    laplace = sp.eye(adj.shape[0]) - normalize_adj(adj)
    if scope_flag == 1:
        scope = torch.load(path + "/scope_1.pt")
    if scope_flag == 2:
        scope = torch.load(path + "/scope_2.pt")
    return adj, feat, label, num_class, idx_train, idx_val, idx_test, laplace, scope


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:2')

    print(args)
    path = "../dataset/" + args.dataname
    adj, feat, labels, num_class, train_idx, val_idx, test_idx, laplace, scope = get_dataset(path, args.dataname,
                                                                                              args.scope_flag)
    adj = adj + sp.eye(adj.shape[0])
    edge_index = coo_to_edge_index(adj)
    ori_g = Graph(x=feat, edge_index=edge_index)

    if args.dataname == 'pubmed':
        new_adjs = []
        for i in range(10):
            new_adjs.append(sp.load_npz(path + "/0.01_1_" + str(i) + ".npz"))
        adj_num = len(new_adjs)
        adj_inter = int(adj_num / args.num)
        sele_adjs = []
        for i in range(args.num + 1):
            try:
                if i == 0:
                    sele_adjs.append(new_adjs[i])
                else:
                    sele_adjs.append(new_adjs[i * adj_inter - 1])
            except IndexError:
                pass
        print("Number of select adjs:", len(sele_adjs))
        epoch_inter = args.epoch_inter
    else:
        scope_matrix = sp.coo_matrix((np.ones(scope.shape[1]), (scope[0, :], scope[1, :])), shape=adj.shape).A
        dist = adj.A.sum(-1) / adj.A.sum()
        # dist边分布
    in_dim = feat.shape[1]

    activation = tlx.nn.PRelu()

    model = DGIModel(in_feat=feat.shape[1], hid_feat=args.hid_dim, act=activation).to(args.device)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    N = feat.shape[0]

    cnt_wait = 0
    best = 1e9
    best_t = 0

    #### SpCo ######
    theta = args.theta
    delta = np.ones(adj.shape) * args.delta_origin  # 两两之间的权值全部初始增强为0.5。
    delta_add = delta
    delta_dele = delta
    num_node = adj.shape[0]
    range_node = np.arange(num_node)
    ori_graph = ori_g  # 原始图
    new_graph = ori_g  # 增强图

    new_adj = adj.tocsc()
    ori_attr = torch.Tensor(new_adj[new_adj.nonzero()])[0]
    ori_diag_attr = torch.Tensor(new_adj[range_node, range_node])[0]
    new_attr = torch.Tensor(new_adj[new_adj.nonzero()])[0]
    new_diag_attr = torch.Tensor(new_adj[range_node, range_node])[0]

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        graph1_, attr1, feat1 = random_aug(new_graph, new_attr, new_diag_attr, feat, args.dfr_1, args.der_1)  # 更新
        graph2_, attr2, feat2 = random_aug(ori_graph, ori_attr, ori_diag_attr, feat, args.dfr_2, args.der_2)  # 不更新

        graph1 = graph1_.edge_index.to(args.device)
        graph2 = graph2_.edge_index.to(args.device)

        attr1 = attr1.to(args.device)
        attr2 = attr2.to(args.device)

        feat1 = feat1.to(args.device)
        feat2 = feat2.to(args.device)
        loss1 = model(feat1, graph1, attr1, feat2, graph2, attr2)

        if torch.isnan(loss1) == True:
            break

        if loss1 < best:
            best = loss1
            best_t = epoch
            cnt_wait = 0
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

        loss1.backward()
        optimizer.step()

        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss1.item()))
        if args.dataname == 'pubmed':
            if (epoch - 1) % epoch_inter == 0:
                try:
                    print("================================================")
                    delta = args.lam * sele_adjs[int(epoch / epoch_inter)]
                    new_adj = adj + delta

                    new_edge_index = coo_to_edge_index(new_adj)
                    new_graph = Graph(edge_index=new_edge_index)  # 更新图
                    new_attr = torch.Tensor(new_adj[new_adj.nonzero()])[0]
                    new_diag_attr = torch.Tensor(new_adj[range_node, range_node])[0]
                except IndexError:
                    pass
        else:
            if epoch % args.turn == 0:
                print("================================================")
                if args.dataname in ["cora", "citeseer"] and epoch != 0:
                    delta_add, delta_dele = plug(theta, laplace, delta_add, delta_dele, args.epsilon, dist,
                                                 args.sin_iter, True)
                else:
                    delta_add, delta_dele = plug(theta, laplace, delta_add, delta_dele, args.epsilon, dist,
                                                 args.sin_iter)
                delta = (delta_add - delta_dele) * scope_matrix
                delta = args.lam * normalize_adj(delta)
                new_adj = adj + delta

                new_edge_index = coo_to_edge_index(new_adj)
                new_graph = Graph(edge_index=new_edge_index)  # 更新图
                new_attr = torch.Tensor(new_adj[new_adj.nonzero()])[0]  # 更新边权
                new_diag_attr = torch.Tensor(new_adj[range_node, range_node])[0]  # 更新边权
                theta = update(1, epoch, args.epochs)

    print("=== Evaluation ===")
    ori_edge_index = ori_g.edge_index.to(args.device)
    feat = feat.to(args.device)

    attr = torch.Tensor(adj[adj.nonzero()])[0].to(args.device)
    embeds = model.get_embedding(feat, ori_edge_index, attr)
    test_f1_macro_ll = 0
    test_f1_micro_ll = 0

    label_dict = {0: "5", 1: "10", 2: "20"}
    for i in range(3):
        train_embs = embeds[train_idx[i]]
        val_embs = embeds[val_idx]
        test_embs = embeds[test_idx]

        label = labels.to(args.device)

        train_labels = label[train_idx[i]]
        val_labels = label[val_idx]
        test_labels = label[test_idx]

        ''' Linear Evaluation '''
        logreg = LogReg(train_embs.shape[1], num_class)
        opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)
        logreg = logreg.to(args.device)

        loss_fn = th.nn.CrossEntropyLoss()

        best_val_acc = 0
        eval_acc = 0

        for epoch in range(2000):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            preds = th.argmax(logits, dim=1)
            train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
            loss2 = loss_fn(logits, train_labels)
            loss2.backward()
            opt.step()

            logreg.eval()
            with th.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)

                val_preds = th.argmax(val_logits, dim=1)
                test_preds = th.argmax(test_logits, dim=1)

                val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
                test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]

                test_f1_macro = f1_score(test_labels.cpu(), test_preds.cpu(), average='macro')
                test_f1_micro = f1_score(test_labels.cpu(), test_preds.cpu(), average='micro')
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    if test_acc > eval_acc:
                        test_f1_macro_ll = test_f1_macro
                        test_f1_micro_ll = test_f1_micro

                print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc,
                                                                                         test_acc))

        f = open(own_str + "_" + label_dict[i] + ".txt", "a")
        f.write(str(test_f1_macro_ll) + "\t" + str(test_f1_micro_ll) + "\n")
        f.close()
