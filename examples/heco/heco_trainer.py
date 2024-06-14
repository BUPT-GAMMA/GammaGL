# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   HeCo_trainer.py
@Time    :   
@Author  :   tan jiarui
"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR
#To run this, you should change line 186, String'YourFileDirectory' to your own file directory
import argparse
import numpy as np
import scipy.sparse as sp
import tensorlayerx as tlx
import tensorlayerx.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from gammagl.models.heco import HeCo
from tensorlayerx.model import WithLoss
from gammagl.datasets.acm4heco import ACM4HeCo
import scipy.sparse as sp
#Mention: all 'str' in this code should be replaced with your own file directories
class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, W_init='he_normal'),
            nn.ELU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, W_init='he_normal')
        )
        self.tau = tau
        self.lam = lam
    def sim(self, z1, z2):
        z1_norm = tlx.l2_normalize(z1, axis=-1)
        z2_norm = tlx.l2_normalize(z2, axis=-1)
        z1_norm = tlx.reshape(tlx.reduce_mean(z1/z1_norm, axis=-1), (-1, 1))
        z2_norm = tlx.reshape(tlx.reduce_mean(z2/z2_norm, axis=-1), (-1, 1))
        dot_numerator = tlx.matmul(z1, tlx.transpose(z2))
        dot_denominator = tlx.matmul(z1_norm, tlx.transpose(z2_norm))
        sim_matrix = tlx.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self , z, pos):
        z_mp = z.get("z_mp")
        z_sc = z.get("z_sc")
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = tlx.transpose(matrix_mp2sc)
        
        matrix_mp2sc = matrix_mp2sc / (tlx.reshape(tlx.reduce_sum(matrix_mp2sc, axis=1), (-1, 1)) + 1e-8)
        lori_mp = -tlx.reduce_mean(tlx.log(tlx.reduce_sum(tlx.multiply(matrix_mp2sc, pos), axis=-1)))

        matrix_sc2mp = matrix_sc2mp / (tlx.reshape(tlx.reduce_sum(matrix_sc2mp, axis=1), (-1, 1)) + 1e-8)
        lori_sc = -tlx.reduce_mean(tlx.log(tlx.reduce_sum(tlx.multiply(matrix_sc2mp, pos), axis=-1)))
        return self.lam * lori_mp + (1 - self.lam) * lori_sc

class Contrast_Loss(WithLoss):
    def __init__(self, net, loss_fn):
        super(Contrast_Loss, self).__init__(backbone=net, loss_fn=loss_fn)
    
    def forward(self, datas, pos):
        z = self.backbone_network(datas)
        loss = self._loss_fn(z, pos)
        return loss

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(in_features=ft_in, out_features=nb_classes, W_init='xavier_uniform', b_init='constant')

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


def evaluate(embeds, idx_train, idx_val, idx_test, label, nb_classes, lr, wd, isTest=True):
    hid_units = tlx.get_tensor_shape(embeds)[1]
    train_embs_list = []
    val_embs_list = []
    test_embs_list = []
    label_train_list = []
    label_val_list = []
    label_test_list = []
    for i in range(0, len(idx_train)):
        train_embs_list.append(embeds[idx_train[i]])
        label_train_list.append(label[idx_train[i]])
    for i in range(0, len(idx_val)):
        val_embs_list.append(embeds[idx_val[i]])
        label_val_list.append(label[idx_val[i]])
    for i in range(0, len(idx_test)):
        test_embs_list.append(embeds[idx_test[i]])
        label_test_list.append(label[idx_test[i]])
    train_embs = tlx.stack(train_embs_list, axis = 0)
    val_embs = tlx.stack(val_embs_list, axis = 0)
    test_embs = tlx.stack(test_embs_list, axis = 0)
    label_train = tlx.stack(label_train_list, axis = 0)
    label_val = tlx.stack(label_val_list, axis = 0)
    label_test = tlx.stack(label_test_list, axis = 0)
    train_lbls_idx = tlx.argmax(label_train, axis=-1)
    val_lbls_idx = tlx.argmax(label_val, axis=-1)
    test_lbls_idx = tlx.argmax(label_test, axis=-1)
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []
    #this is the training process for pytorch and paddle(all are recommended)
    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
            #print(lr)
        optimizer = tlx.optimizers.Adam(lr=lr, weight_decay=float(wd)) #Adam method
        loss = tlx.losses.softmax_cross_entropy_with_logits
        log_with_loss = tlx.model.WithLoss(log, loss)
        train_one_step = tlx.model.TrainOneStep(log_with_loss, optimizer, log.trainable_weights)
        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []
        logits_list = []
        for iter_ in range(400):#set this parameter: 'acm'=400
            log.set_train()
            train_one_step(train_embs, train_lbls_idx)
            logits = log(val_embs)
            preds = tlx.argmax(logits, axis = 1) 
            acc_val = 0
            for i in range(0, len(val_lbls_idx)):
                if(preds[i] == val_lbls_idx[i]):
                    acc_val = acc_val + 1
            val_acc = acc_val/len(val_lbls_idx)
            val_f1_macro = f1_score(val_lbls_idx.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls_idx.cpu(), preds.cpu(), average='micro')
            val_accs.append(val_acc)
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)
            logits = log(test_embs)
            preds = tlx.argmax(logits, axis=1)
            acc_test = 0
            for i in range(0, len(test_lbls_idx)):
                if(preds[i] == test_lbls_idx[i]):
                    acc_test = acc_test + 1
            test_acc = acc_test/len(test_lbls_idx)
            test_f1_macro = f1_score(test_lbls_idx.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls_idx.cpu(), preds.cpu(), average='micro')
            test_accs.append(test_acc)
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)
        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])
        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])
        best_logits = logits_list[max_iter]
        best_proba = tlx.softmax(best_logits, axis=1)
        auc_score_list.append(roc_auc_score(y_true=tlx.convert_to_numpy(test_lbls_idx),
                                            y_score=tlx.convert_to_numpy(best_proba),
                                            multi_class='ovr'
                                        ))
    if isTest:
        print("\t[Classification] Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc: {:.4f} "
              .format(np.mean(macro_f1s),
                      np.std(macro_f1s),
                      np.mean(micro_f1s),
                      np.std(micro_f1s),
                      np.mean(auc_score_list),
                      np.std(auc_score_list)
                      )
              )
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)

def main(args):
    dataset = ACM4HeCo(root = 'YourFileDirectory')
    graph = dataset[0]
    edge_pa = graph['edge_index_dict'][('paper', 'to', 'author')]
    edge_ps = graph['edge_index_dict'][('paper', 'to', 'subject')]
    feats = []
    feats.append(graph['paper'].x)
    feats.append(graph['author'].x)
    feats.append(graph['subject'].x)
    mps = graph['metapath']
    pos = graph['pos_set_for_contrast']
    label = graph['paper'].y
    idx_train = graph['train']
    idx_val = graph['val']
    idx_test = graph['test']
    nei_num = graph['nei_num']
    pa_ar = tlx.convert_to_numpy(edge_pa)
    p_list = pa_ar[0]
    a_list = pa_ar[1]
    edge_pa = sp.coo_matrix((np.ones(len(p_list)), (np.array(p_list), np.array(a_list))), shape=(4019, 7167)).toarray()
    p_n_a = []
    for i in range(0, 4019):
        row = edge_pa[i]
        p_a = []
        for j in range(0, 7167):
            if(row[j]) != 0:
                p_a.append(j)
        p_n_a.append(p_a)

    p_n_a = [np.array(i) for i in p_n_a]
    nei_a = p_n_a
    nei_a = [tlx.convert_to_tensor(i, 'int64') for i in nei_a]

    ps_ar = tlx.convert_to_numpy(edge_ps)
    p_list = ps_ar[0]
    s_list = ps_ar[1]
    edge_ps = sp.coo_matrix((np.ones(len(p_list)), (np.array(p_list), np.array(s_list))), shape=(4019, 60)).toarray()
    p_n_s = []
    i = 0
    for i in range(0, 4019):
        row = edge_ps[i]
        p_s = []
        for j in range(0, 60):
            if(row[j]) != 0:
                p_s.append(j)
        p_n_s.append(p_s)
    p_n_s = [np.array(i) for i in p_n_s]
    nei_s = p_n_s
    nei_s = [tlx.convert_to_tensor(i, 'int64') for i in nei_s]
    nei_index = []
    nei_index.append(nei_a)
    nei_index.append(nei_s)
    datas = {
        "feats": feats,
        "mps": mps,
        "nei_index": nei_index,
    }
    n_classes = tlx.get_tensor_shape(label)[1]
    feats_dim_list = [tlx.get_tensor_shape(i)[1] for i in feats]
    P = int(len(mps))
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", P)

    model = HeCo(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                    P, args.sample_rate, nei_num)
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    contrast_loss = Contrast(args.hidden_dim, args.tau, args.lam)
    best_t = 0
    best = 1e9
    loss_func = Contrast_Loss(model, contrast_loss)
    weights_to_train = model.trainable_weights+contrast_loss.trainable_weights
    train_one_step = tlx.model.TrainOneStep(loss_func, optimizer, weights_to_train)
    for epoch in range(args.n_epochs):
        loss = train_one_step(datas, pos)
        print("Epoch:", epoch)
        print("train_loss:", loss)
        if loss < best:
            best = loss
            best_t = epoch
            model.save_weights(model.name+".npz", format='npz_dict')
    print('Loading {}th epoch'.format(best_t))
    model.load_weights(model.name+".npz", format='npz_dict')
    model.set_eval()
    embeds = model.get_embeds(feats, mps)
    # To evaluate the HeCo model with different numbers of training labels, that is 20,40 and 60, which is indicated in the essay of HeCo
    for i in range(len(idx_train)):
        evaluate(embeds, idx_train[i], idx_val[i], idx_test[i], label, n_classes, args.eva_lr, args.eva_wd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="acm", help="your dataset")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60], help="traing ratio")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64, help="hidden dimension")
    parser.add_argument('--n_epochs', type=int, default=10000, help="training epoches for heco")
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05, help="learning rate for evaluate approach")
    parser.add_argument('--eva_wd', type=float, default=0, help="weight decay for evaluate approach")
    
    # The parameters of learning process
    parser.add_argument('--lr', type=float, default=0.0075, help="learing rate for training model")  # 0.0008
    parser.add_argument('--l2_coef', type=float, default=0.0, help="L2 parameter")
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8, help="tau parameter for contrast learning")
    parser.add_argument('--feat_drop', type=float, default=0.3, help="dropping rate for feature")
    parser.add_argument('--attn_drop', type=float, default=0.5, help="dropping rate for attention layer")
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[7, 1], help="sample number for network schema encoding")
    parser.add_argument('--lam', type=float, default=0.5, help="lam parameter for contrast learning")
    args = parser.parse_args()

    main(args)
