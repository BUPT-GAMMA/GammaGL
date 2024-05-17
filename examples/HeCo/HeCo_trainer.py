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
import numpy
import random
import argparse
import warnings
import numpy as np
import scipy.sparse as sp
import tensorlayerx as tlx
import tensorlayerx.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from gammagl.models.HeCo import HeCo
from gammagl.models.contrast_learningHeCo import Contrast
from gammagl.datasets.ACM_data_process import ACM4HeCo
import scipy.sparse as sp
#Mention: all 'str' in this code should be replaced with your own file directories


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(in_features=ft_in, out_features=nb_classes, W_init='xavier_uniform', b_init='constant')

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


def evaluate(embeds, ratio, idx_train, idx_val, idx_test, label, nb_classes, dataset, lr, wd, isTest=True):
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
        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []
        logits_list = []
        for iter_ in range(200):#set this parameter: 'acm'=200
            log.set_train()
            log_with_loss = tlx.model.WithLoss(log, loss)
            train_one_step = tlx.model.TrainOneStep(log_with_loss, optimizer, log.trainable_weights)
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
    data_load = ACM4HeCo("str", "acm") # str is your file directory
    data_load.download()
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test = data_load.data4HeCo(args)
    isTest=True
    datas = {
        "feats": feats,
        "mps": mps,
        "nei_index": nei_index,
    }
    nb_classes = tlx.get_tensor_shape(label)[1]
    feats_dim_list = [tlx.get_tensor_shape(i)[1] for i in feats]
    P = int(len(mps))
    print("seed ",args.seed)
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", P)

    model = HeCo(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                    P, args.sample_rate, args.nei_num)
    optimizer = tlx.optimizers.Adam(lr=0.008, weight_decay=args.l2_coef)
    contrast_loss = Contrast(args.hidden_dim, args.tau, args.lam)
    cnt_wait = 0
    best = 1e9
    best_t = 0
    cnt = 0
    for epoch in range(args.nb_epochs):  #args.nb_epochs
        model_with_loss = tlx.model.WithLoss(model, contrast_loss)
        weights_to_train = model.trainable_weights+contrast_loss.trainable_weights
        train_one_step = tlx.model.TrainOneStep(model_with_loss, optimizer, weights_to_train)
            
        loss = train_one_step(datas, pos)
        print("loss ", loss)
        '''loss = model(datas, pos)
        grads_in = optimizer.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads_in, model.trainable_weights))
            print("loss ", loss)'''


            # use your file directory to remove 'str', the files to install are '.npz' files
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            model.save_weights('str', format='npz_dict')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
    print('Loading {}th epoch'.format(best_t))
    model.load_weights('str', format='npz_dict')
    model.set_eval()
    os.remove('str')
    embeds = model.get_embeds(feats, mps)
    # To evaluate the HeCo model with different numbers of training labels, that is 20,40 and 60, which is indicated in the essay of HeCo
    for i in range(len(idx_train)):
        evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, args.dataset, args.eva_lr, args.eva_wd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning processz
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)  # 0.0008
    parser.add_argument('--l2_coef', type=float, default=0.0)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[7, 1])
    parser.add_argument('--lam', type=float, default=0.5)
    
    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    own_str = args.dataset

    main(args)
    
