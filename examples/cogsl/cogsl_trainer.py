import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR


import sys
import argparse
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity as cos
from scipy.linalg import fractional_matrix_power, inv

import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.datasets import WikiCS, PolBlogs, Planetoid, Coauthor
from gammagl.models import CoGSLModel
from gammagl.utils import  mask_to_index, set_device, add_self_loops
from tensorlayerx.model import TrainOneStep, WithLoss


class VeLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(VeLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.net = net

    def forward(self, data, y):
        new_v1, new_v2 = self.net.get_view(data)
        logits_v1, logits_v2, prob_v1, prob_v2 = self.net.get_cls_loss(new_v1, new_v2, data['x'])
        curr_v = self.net.get_fusion(new_v1, prob_v1, new_v2, prob_v2)
        logits_v = self.net.get_v_cls_loss(curr_v, data['x'])

        views = [curr_v, new_v1, new_v2]

        loss_v1 = self._loss_fn(tlx.gather(logits_v1, data['train_idx']), tlx.gather(data['y'], data['train_idx']))
        loss_v2 = self._loss_fn(tlx.gather(logits_v2, data['train_idx']), tlx.gather(data['y'], data['train_idx']))
        loss_v = self._loss_fn(tlx.gather(logits_v, data['train_idx']), tlx.gather(data['y'], data['train_idx']))

        cls_loss = args.cls_coe * loss_v + (loss_v1 + loss_v2) * (1 - args.cls_coe) / 2

        vv1, vv2, v1v2 = self.net.get_mi_loss(data['x'], views)
        mi_loss = args.mi_coe * v1v2 + (vv1 + vv2) * (1 - args.mi_coe) / 2
        loss = cls_loss - data['curr'] * mi_loss
        return loss

class MiLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(MiLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.net = net

    def forward(self, data, y):
        new_v1, new_v2 = self.net.get_view(data)
        logits_v1, logits_v2, prob_v1, prob_v2 = self.net.get_cls_loss(new_v1, new_v2, data['x'])
        curr_v = self.net.get_fusion(new_v1, prob_v1, new_v2, prob_v2)

        views = [curr_v, new_v1, new_v2]

        vv1, vv2, v1v2 = self.net.get_mi_loss(data['x'], views)
        loss = args.mi_coe * v1v2 + (vv1 + vv2) * (1 - args.mi_coe) / 2
        return loss

class ClsLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(ClsLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.net = net

    def forward(self, data, y):
        new_v1, new_v2 = self.net.get_view(data)
        logits_v1, logits_v2, prob_v1, prob_v2 = self.net.get_cls_loss(new_v1, new_v2, data['x'])
        curr_v = self.net.get_fusion(new_v1, prob_v1, new_v2, prob_v2)
        logits_v = self.net.get_v_cls_loss(curr_v, data['x'])

        loss_v1 = self._loss_fn(tlx.gather(logits_v1, data['train_idx']), tlx.gather(data['y'], data['train_idx']))
        loss_v2 = self._loss_fn(tlx.gather(logits_v2, data['train_idx']), tlx.gather(data['y'], data['train_idx']))
        loss_v = self._loss_fn(tlx.gather(logits_v, data['train_idx']), tlx.gather(data['y'], data['train_idx']))
        loss = args.cls_coe * loss_v + (loss_v1 + loss_v2) * (1 - args.cls_coe) / 2
        return loss

def gen_auc_mima(logits, label):
    preds = tlx.argmax(logits, axis=1)
    test_f1_macro = f1_score(label.cpu(), preds.cpu(), average='macro')
    test_f1_micro = f1_score(label.cpu(), preds.cpu(), average='micro')

    best_proba = nn.Softmax(axis=1)(logits)
    if logits.shape[1] != 2:
        auc = roc_auc_score(y_true=label.detach().cpu().numpy(),
                            y_score=best_proba.detach().cpu().numpy(),
                            multi_class='ovr'
                            )
    else:
        auc = roc_auc_score(y_true=label.detach().cpu().numpy(),
                            y_score=best_proba[:, 1].detach().cpu().numpy()
                            )
    return test_f1_macro, test_f1_micro, auc

def accuracy(output, label):
    preds = tlx.argmax(output, axis=1)
    if tlx.BACKEND == 'torch':
        correct = preds.eq(label).double()
    else:
        correct = tlx.convert_to_tensor(tlx.equal(preds, label), dtype=tlx.float32)
    correct = correct.sum()
    return correct / len(label)

def loss_acc(output, y):
    loss = tlx.losses.softmax_cross_entropy_with_logits(output, y)
    acc = accuracy(output, y)
    return loss, acc

def train_cls(main_model, data):
    new_v1, new_v2 = main_model.get_view(data)
    logits_v1, logits_v2, prob_v1, prob_v2 = main_model.get_cls_loss(new_v1, new_v2, data['x'])
    curr_v = main_model.get_fusion(new_v1, prob_v1, new_v2, prob_v2)
    logits_v = main_model.get_v_cls_loss(curr_v, data['x'])

    views = [curr_v, new_v1, new_v2]

    loss_v1, _ = loss_acc(logits_v1[data['train_idx']], data['y'][data['train_idx']])
    loss_v2, _ = loss_acc(logits_v2[data['train_idx']], data['y'][data['train_idx']])
    loss_v, _ = loss_acc(logits_v[data['train_idx']], data['y'][data['train_idx']])
    return args.cls_coe * loss_v + (loss_v1 + loss_v2) * (1 - args.cls_coe) / 2, views

def get_khop_indices(k, view):
    view = sp.csr_matrix((view.A > 0).astype("int32"))
    view_ = view
    for i in range(1, k):
        view_ = sp.csr_matrix((np.matmul(view_.toarray(), view.T.toarray()) > 0).astype("int32"))
    return tlx.convert_to_tensor(view_.nonzero())

def topk(k, adj):
    adj = adj.toarray()
    pos = np.zeros(adj.shape)
    for i in range(adj.shape[0]):
      one = adj[i].nonzero()[0]
      if len(one)>k:
        oo = np.argsort(-adj[i, one])
        sele = one[oo[:k]]
        pos[i, sele] = adj[i, sele]
      else:
        pos[i, one] = adj[i, one]
    return pos

def knn(feat, num_node, k):
    adj = np.zeros((num_node, num_node), dtype=np.int64)
    dist = cos(feat)
    col = np.argpartition(dist, -(k + 1), axis=1)[:, -(k + 1):].flatten()
    adj[np.arange(num_node).repeat(k + 1), col] = 1
    adj = sp.coo_matrix(adj)
    return adj

def diff(adj, alpha):
    d = np.diag(np.array(np.sum(adj, 1)).flatten())
    dinv = fractional_matrix_power(d, -0.5)
    at = np.matmul(np.matmul(dinv, adj.toarray()), dinv)
    at[np.isnan(at)] = 0
    at[np.isinf(at)] = 0
    adj = alpha * inv((np.eye(adj.shape[0]) - (1 - alpha) * at))
    adj = sp.coo_matrix(adj)
    return adj

def main(args):
    if tlx.BACKEND == 'torch' and args.gpu >= 0:
        set_device(int(args.gpu))

    if args.dataset == 'citeseer':
        dataset = Planetoid('', args.dataset)
    elif args.dataset == 'polblogs':
        dataset = PolBlogs('')
    elif args.dataset == 'wikics':
        dataset = WikiCS('')
    elif args.dataset == 'ms':
        dataset = Coauthor('', 'cs')
    graph = dataset.data
    if args.dataset == 'polblogs':
        arr = np.arange(1490)
        random_indices = np.random.permutation(len(arr))
        train_idx = tlx.convert_to_tensor(arr[random_indices[:121]])
        val_idx = tlx.convert_to_tensor(arr[random_indices[121:224]])
        test_idx = tlx.convert_to_tensor(arr[random_indices[244:1490]])
    elif args.dataset == 'ms':
        arr = np.arange(18333)
        random_indices = np.random.permutation(len(arr))
        train_idx = tlx.convert_to_tensor(arr[random_indices[:300]])
        val_idx = tlx.convert_to_tensor(arr[random_indices[300:800]])
        test_idx = tlx.convert_to_tensor(arr[random_indices[800:1800]])
    else:
        train_idx = mask_to_index(graph.train_mask)
        test_idx = mask_to_index(graph.test_mask)
        val_idx = mask_to_index(graph.val_mask)
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes, n_loops=1)
    value = np.ones(edge_index[0].shape)
    view1 = sp.coo_matrix((value, (edge_index.cpu())), shape=(graph.num_nodes, graph.num_nodes))
    v1_indice = get_khop_indices(args.v1_p, view1)
    if args.dataset == 'wikics' or args.dataset == 'ms':
        view2 = view1
        v2_indice = get_khop_indices(1, view2)
    else:
        view2 = diff(view1, 0.1)
        kn = topk(args.v2_p, view2)
        kn = sp.coo_matrix(kn)
        v2_indice = get_khop_indices(1, kn)
    view1 = tlx.convert_to_tensor(view1.todense(), dtype=tlx.float32)
    view2 = tlx.convert_to_tensor(view2.todense(), dtype=tlx.float32)
    v1_indice = tlx.convert_to_tensor(v1_indice, dtype=tlx.int64)
    v2_indice = tlx.convert_to_tensor(v2_indice, dtype=tlx.int64)

    net = CoGSLModel(dataset.num_node_features, args.cls_hid, dataset.num_classes,
                                     args.gen_hid, args.mi_hid, args.com_lambda_v1, args.com_lambda_v2,
                           args.lam, args.alpha, args.cls_dropout, args.ve_dropout, args.tau, args.ggl, args.big, args.batch)

    scheduler = tlx.optimizers.lr.ExponentialDecay(learning_rate=args.ve_lr, gamma=0.99)
    opti_ve = tlx.optimizers.Adam(lr=scheduler, weight_decay=args.ve_weight_decay)
    opti_cls = tlx.optimizers.Adam(lr=args.cls_lr, weight_decay=args.cls_weight_decay)
    opti_mi = tlx.optimizers.Adam(lr=args.mi_lr, weight_decay=args.mi_weight_decay)

    ve_train_weights = net.ve.trainable_weights
    cls_train_weights = net.cls.trainable_weights
    mi_train_weights = net.mi.trainable_weights

    ve_loss = VeLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    ve_train_one_step = TrainOneStep(ve_loss, opti_ve, ve_train_weights)

    cls_loss = ClsLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    cls_train_one_step = TrainOneStep(cls_loss, opti_cls, cls_train_weights)

    mi_loss = MiLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    mi_train_one_step = TrainOneStep(mi_loss, opti_mi, mi_train_weights)
    data = {
        "name" : args.dataset,
        "x": graph.x,
        "y": graph.y,
        "edge_index": graph.edge_index,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "view1": view1,
        "view2": view2,
        "v1_indice": v1_indice,
        "v2_indice": v2_indice,
        "num_nodes": graph.num_nodes,
        "curr": 0,
    }

    best_acc_val = 0
    best_loss_val = 1e9
    best_v = None

    for epoch in range(args.n_epoch):
        curr = np.log(1 + args.temp_r * epoch)
        curr = min(max(0.05, curr), 0.1)
        data['curr'] = curr

        net.set_train()
        for inner_ve in range(args.inner_ve_epoch):
            ve_loss=ve_train_one_step(data, graph.y)
            # print('ve_loss=', ve_loss)

        for inner_cls in range(args.inner_cls_epoch):
            cls_loss=cls_train_one_step(data, graph.y)
            # print('cls_loss=',  cls_loss)

        for inner_mi in range(args.inner_mi_epoch):
            mi_loss=mi_train_one_step(data, graph.y)
            # print('mi_loss=', mi_loss)

        ## validation ##
        net.set_eval()
        _, views = train_cls(net, data)
        logits_v_val = net.get_v_cls_loss(views[0], data['x'])
        loss_val, acc_val = loss_acc(logits_v_val[data['val_idx']], data['y'][data['val_idx']])
        if acc_val>best_acc_val or (acc_val == best_acc_val and best_loss_val > loss_val):
            print("better v!")
            best_acc_val = max(acc_val, best_acc_val)
            best_loss_val = loss_val
            net.cls.encoder_v.save_weights(data['name'] + ".npz", format='npz_dict')
            best_v = views[0]
        print("EPOCH ", epoch, "\tCUR_LOSS_VAL ", loss_val.item(), "\tCUR_ACC_Val ",
              acc_val.item(), "\tBEST_ACC_VAL ", best_acc_val.item())

    ## test ##
    net.cls.encoder_v.load_weights(data['name'] + ".npz", format='npz_dict')
    net.set_eval()

    probs = net.cls.encoder_v(data['x'], best_v)
    test_f1_macro, test_f1_micro, auc = gen_auc_mima(probs[data['test_idx']], data['y'][data['test_idx']])
    print("Test_Macro: ", test_f1_macro, "\tTest_Micro: ", test_f1_micro, "\tAUC: ", auc)

    f = open(f'{tlx.BACKEND}_results/{args.dataset}' + ".txt", "a")
    f.write("v1_p=" + str(args.v1_p) + "\t" + "v2_p=" + str(args.v2_p) + "\t" + str(test_f1_macro) + "\t" + str(test_f1_micro) + "\t" + str(auc) + "\n")
    f.close()

def polblogs_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="polblogs")
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--v1_p', type=int, default=1)
    parser.add_argument('--v2_p', type=int, default=300)
    parser.add_argument('--cls_hid', type=int, default=16)
    ## gen
    parser.add_argument('--com_lambda_v1', type=float, default=0.1)
    parser.add_argument('--com_lambda_v2', type=float, default=1.0)
    parser.add_argument('--gen_hid', type=int, default=64)
    ## fusion
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1)
    ## mi
    parser.add_argument('--mi_hid', type=int, default=128)
    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)
    parser.add_argument('--ve_lr', type=float, default=0.1)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    parser.add_argument('--ve_dropout', type=float, default=0.8)
    parser.add_argument('--mi_lr', type=float, default=0.01)
    parser.add_argument('--mi_weight_decay', type=float, default=0.)
    ## iter
    parser.add_argument('--n_epoch', type=int, default=150)
    parser.add_argument('--inner_ve_epoch', type=int, default=1)
    parser.add_argument('--inner_cls_epoch', type=int, default=5)
    parser.add_argument('--inner_mi_epoch', type=int, default=5)
    parser.add_argument('--temp_r', type=float, default=1e-4)
    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)
    parser.add_argument('--mi_coe', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=0.8)
    #####################################

    args, _ = parser.parse_known_args()
    return args


def citeseer_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="citeseer")
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--v1_p', type=int, default=2)
    parser.add_argument('--v2_p', type=int, default=40)
    parser.add_argument('--cls_hid', type=int, default=16)
    ## gen
    parser.add_argument('--com_lambda_v1', type=float, default=0.1)
    parser.add_argument('--com_lambda_v2', type=float, default=0.1)
    parser.add_argument('--gen_hid', type=int, default=32)
    ## fusion
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.1)
    ## mi
    parser.add_argument('--mi_hid', type=int, default=128)
    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)
    parser.add_argument('--ve_lr', type=float, default=0.001)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    parser.add_argument('--ve_dropout', type=float, default=0.5)
    parser.add_argument('--mi_lr', type=float, default=0.01)
    parser.add_argument('--mi_weight_decay', type=float, default=0.)
    ## iter
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--inner_ve_epoch', type=int, default=5)
    parser.add_argument('--inner_cls_epoch', type=int, default=5)
    parser.add_argument('--inner_mi_epoch', type=int, default=10)
    parser.add_argument('--temp_r', type=float, default=1e-4)
    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)
    parser.add_argument('--mi_coe', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=0.8)
    #####################################

    args, _ = parser.parse_known_args()
    return args


def wikics_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="wikics")
    parser.add_argument('--batch', type=int, default=4000)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--v1_p', type=int, default=1)
    parser.add_argument('--v2_p', type=int, default=1)
    parser.add_argument('--cls_hid', type=int, default=16)
    ## gen
    parser.add_argument('--com_lambda_v1', type=float, default=0.5)
    parser.add_argument('--com_lambda_v2', type=float, default=0.5)
    parser.add_argument('--gen_hid', type=int, default=16)
    ## fusion
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1)
    ## mi
    parser.add_argument('--mi_hid', type=int, default=32)
    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)
    parser.add_argument('--ve_lr', type=float, default=0.01)
    parser.add_argument('--ve_weight_decay', type=float, default=0)
    parser.add_argument('--ve_dropout', type=float, default=0.2)
    parser.add_argument('--mi_lr', type=float, default=0.01)
    parser.add_argument('--mi_weight_decay', type=float, default=0.)
    ## iter
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--inner_ve_epoch', type=int, default=1)
    parser.add_argument('--inner_cls_epoch', type=int, default=1)
    parser.add_argument('--inner_mi_epoch', type=int, default=1)
    parser.add_argument('--temp_r', type=float, default=1e-3)
    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)
    parser.add_argument('--mi_coe', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=0.5)
    #####################################

    args, _ = parser.parse_known_args()
    return args


def ms_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="ms")
    parser.add_argument('--batch', type=int, default=1000)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--v1_p', type=int, default=1)
    parser.add_argument('--v2_p', type=int, default=1)
    parser.add_argument('--cls_hid', type=int, default=16)
    ## gen
    parser.add_argument('--com_lambda_v1', type=float, default=0.5)
    parser.add_argument('--com_lambda_v2', type=float, default=0.5)
    parser.add_argument('--gen_hid', type=int, default=32)
    ## fusion
    parser.add_argument('--lam', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=1.0)
    ## mi
    parser.add_argument('--mi_hid', type=int, default=256)
    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)
    parser.add_argument('--ve_lr', type=float, default=0.0001)
    parser.add_argument('--ve_weight_decay', type=float, default=1e-10)
    parser.add_argument('--ve_dropout', type=float, default=0.8)
    parser.add_argument('--mi_lr', type=float, default=0.01)
    parser.add_argument('--mi_weight_decay', type=float, default=0.)
    ## iter
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--inner_ve_epoch', type=int, default=1)
    parser.add_argument('--inner_cls_epoch', type=int, default=15)
    parser.add_argument('--inner_mi_epoch', type=int, default=10)
    parser.add_argument('--temp_r', type=float, default=1e-4)
    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)
    parser.add_argument('--mi_coe', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=0.5)
    #####################################

    args, _ = parser.parse_known_args()
    return args

argv = sys.argv
dataset = argv[1].split(' ')[0]

def set_params():
    args = polblogs_params()
    if dataset == "polblogs":
        args = polblogs_params()
        args.ggl = False
        args.big = False
    elif dataset == "citeseer":
        args = citeseer_params()
        args.ggl = False
        args.big = False
    elif dataset == "wikics":
        args = wikics_params()
        args.ggl = True
        args.big = True
    elif dataset == "ms":
        args = ms_params()
        args.ggl = True
        args.big = True
    return args


if __name__ == '__main__':
    args = set_params()
    main(args)