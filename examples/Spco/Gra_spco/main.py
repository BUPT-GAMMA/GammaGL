import os

os.environ['TL_BACKEND'] = 'torch'
import argparse

from aug import random_aug
import numpy as np
import tensorlayerx as tlx
import warnings
from gammagl.data import Graph
from dataset import DataSet
from gammagl.utils import add_self_loops
from gammagl.utils.device import set_device

from sklearn.metrics import f1_score
import scipy.sparse as sp
from params import set_params
from model import Model, LogReg

warnings.filterwarnings('ignore')

args = set_params('citeseer')  # Cora/Citeseer/Pubmed/flickr/blog
own_str = args.dataname
print(own_str)


def coo_to_edge_index(adj):
    edge_index = adj.todense().nonzero()
    edge_index = tlx.convert_to_tensor([edge_index[0], edge_index[1]])
    return edge_index


def sinkhorn(K, dist, sin_iter):
    # make the matrix sum to 1
    u = np.ones([len(dist), 1]) / len(dist)  # the distribution of row or col
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


def get_dataset(path, data_name, scope_flag):
    dataset = DataSet(path=path, name=data_name)
    edge_index, feat, labels, num_class, train_mask, val_mask, test_mask = dataset.load_dataset()
    if data_name != 'blog':
        feat = tlx.convert_to_tensor(preprocess_features(feat), dtype=tlx.float32)
    else:
        feat = tlx.convert_to_tensor(feat, dtype=tlx.float32)
    adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0, :], edge_index[1, :])),
                        shape=(feat.shape[0], feat.shape[0]))
    laplace = sp.eye(adj.shape[0]) - normalize_adj(adj)
    if scope_flag == 1:  # 1-hop Sparser operation
        scope = edge_index
    if scope_flag == 2:  # 2-hop Sparser operation
        adj_2 = adj @ adj
        scope = coo_to_edge_index(adj_2)
    return adj, feat, labels, num_class, train_mask, val_mask, test_mask, laplace, scope


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    print(args)
    path = "../datasets"
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    adj, feat, labels, num_class, train_mask, val_mask, test_mask, laplace, scope = get_dataset(path, args.dataname,
                                                                                                args.scope_flag)
    device = set_device(id=args.gpu)

    adj = adj + sp.eye(adj.shape[0])  # add self-loop
    edge_index = coo_to_edge_index(adj)

    feat = feat.to(edge_index.device)
    labels = labels.to(edge_index.device)
    train_mask = train_mask.to(edge_index.device)
    val_mask = val_mask.to(edge_index.device)
    test_mask = test_mask.to(edge_index.device)

    ori_g = Graph(x=feat, edge_index=edge_index)

    if args.dataname == 'pubmed':
        new_adjs = []
        adjs_path = '../pubmed_new_adjs'
        for i in range(10):
            new_adjs.append(sp.load_npz(adjs_path + "/0.01_1_" + str(i) + ".npz"))
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
        dist = adj.A.sum(-1) / adj.A.sum()  # get distribution

    in_dim = feat.shape[1]

    if args.dataname == 'citeseer':
        activation = tlx.nn.PRelu()  # citeseer
    else:
        activation = tlx.relu  # cora,pubmed,flickr,blog

    model = Model(in_dim=feat.shape[1], hidden_dim=args.hid_dim, out_dim=args.out_dim, num_layers=args.n_layers,
                  activation=activation, tau=args.tau)

    optim = tlx.optimizers.Adam(lr=args.lr1, weight_decay=args.wd1)

    N = feat.shape[0]
    cnt_wait = 0
    best = 1e9
    best_t = 0

    #### SpCo ######
    theta = args.theta
    delta = np.ones(adj.shape) * args.delta_origin
    delta_add = delta
    delta_dele = delta
    num_node = adj.shape[0]
    range_node = np.arange(num_node)
    ori_graph = ori_g
    new_graph = ori_g

    new_adj = adj.tocsc()
    ori_attr = tlx.convert_to_tensor(new_adj[new_adj.nonzero()], dtype=tlx.float32)[0]  # get A's all edges' weight
    ori_diag_attr = tlx.convert_to_tensor(new_adj[range_node, range_node], dtype=tlx.float32)[0]  # get A's diag weight
    new_attr = tlx.convert_to_tensor(new_adj[new_adj.nonzero()], dtype=tlx.float32)[0]  # get V's all edges' weight
    new_diag_attr = tlx.convert_to_tensor(new_adj[range_node, range_node], dtype=tlx.float32)[0]  # get V's diag weight

    for epoch in range(args.epochs):
        model.train()

        graph1_, attr1, feat1 = random_aug(new_graph, new_attr, new_diag_attr, feat, args.dfr_1, args.der_1)  # 更新
        graph2_, attr2, feat2 = random_aug(ori_graph, ori_attr, ori_diag_attr, feat, args.dfr_2, args.der_2)  # 不更新

        graph1 = graph1_.edge_index
        graph2 = graph2_.edge_index

        loss1 = model(feat1, graph1, attr1, feat2, graph2, attr2)

        if tlx.is_nan(loss1):
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

        grads = optim.gradient(loss1, model.trainable_weights)
        optim.apply_gradients(zip(grads, model.trainable_weights))

        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss1.item()))
        if args.dataname == 'pubmed':
            if (epoch - 1) % epoch_inter == 0:
                try:
                    print("================================================")
                    delta = args.lam * sele_adjs[int(epoch / epoch_inter)]
                    new_adj = adj + delta

                    new_edge_index = coo_to_edge_index(new_adj)
                    new_graph = Graph(edge_index=new_edge_index)
                    new_attr = tlx.convert_to_tensor(new_adj[new_adj.nonzero()], dtype=tlx.float32)[0]
                    new_diag_attr = tlx.convert_to_tensor(new_adj[range_node, range_node], dtype=tlx.float32)[0]
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
                        new_adj = adj + delta  # get enhanced V

                        new_edge_index = coo_to_edge_index(new_adj)
                        new_graph = Graph(edge_index=new_edge_index)  # update edge_index
                        new_attr = tlx.convert_to_tensor(new_adj[new_adj.nonzero()], dtype=tlx.float32)[
                            0]  # update weight
                        new_diag_attr = tlx.convert_to_tensor(new_adj[range_node, range_node], dtype=tlx.float32)[
                            0]  # update self-loop weight
                        theta = update(1, epoch, args.epochs)

    print("=== Evaluation ===")
    ori_edge_index = ori_g.edge_index
    feat = feat

    attr = tlx.convert_to_tensor(adj[adj.nonzero()], dtype=tlx.float32)[0]
    embeds = model.get_embedding(feat, ori_edge_index, attr)  # get representations
    test_f1_macro_ll = 0
    test_f1_micro_ll = 0

    train_embs = embeds[train_mask, :]
    val_embs = embeds[val_mask, :]
    test_embs = embeds[test_mask, :]

    label = labels

    train_labels = label[train_mask]
    val_labels = label[val_mask]
    test_labels = label[test_mask]

    ''' Linear Evaluation '''
    logreg = LogReg(train_embs.shape[1], num_class)
    opt =tlx.optimizers.Adam(lr=args.lr2, weight_decay=args.wd2)

    loss_fn = tlx.losses.softmax_cross_entropy_with_logits

    best_val_acc = 0
    eval_acc = 0

    for epoch in range(2000):
        logreg.train()

        logits = logreg(train_embs)
        preds = logits.argmax(1)
        train_acc = (preds == train_labels).sum().float() / train_labels.shape[0]
        loss = loss_fn(logits, train_labels)

        grads = opt.gradient(loss, logreg.trainable_weights)
        opt.apply_gradients(zip(grads, logreg.trainable_weights))

        logreg.eval()
        val_logits = logreg(val_embs)
        test_logits = logreg(test_embs)

        val_preds = val_logits.argmax(1)
        test_preds = test_logits.argmax(1)

        val_acc = (val_preds == val_labels).sum().float() / val_labels.shape[0]
        test_acc = (test_preds == test_labels).sum().float() / test_labels.shape[0]

        test_f1_macro = f1_score(test_labels.cpu(), test_preds.cpu(), average='macro')
        test_f1_micro = f1_score(test_labels.cpu(), test_preds.cpu(), average='micro')
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            if test_acc > eval_acc:
                test_f1_macro_ll = test_f1_macro
                test_f1_micro_ll = test_f1_micro

        print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc,
                                                                                 test_acc))
    f = open(own_str + "_result" + ".txt", "a")
    f.write(str(test_f1_macro_ll) + "\t" + str(test_f1_micro_ll) + "\n")
    f.close()
