
from gammagl.datasets import HGBDataset, IMDB
import scipy.sparse as sp
import json
import ssl
import urllib.request
import sys
import os
import os.path as osp
from collections import Counter
from typing import Callable, List, Optional
import numpy as np
import tensorlayerx as tlx
import shutil
import errno
from gammagl.data import (HeteroGraph, InMemoryDataset, extract_zip)
from sklearn.preprocessing import OneHotEncoder
import requests
import io
import zipfile
from gammagl.data import download_url
class ACM4HeCo(object):
    def __init__(self, root: str = None, name: str = 'acm'):
        self.name = name
        self.dir = root
        self.url = 'https://github.com/liun-online/HeCo/raw/main/data/acm'
        





    def download(self):
        print("---  download data  ---")
        names = ['pa.txt', 'ps.txt', 'labels.npy', 'p_feat.npz', 'train_20.npy', 'train_40.npy', 'train_60.npy', 'test_20.npy', 'test_40.npy', 'test_60.npy', 'val_20.npy', 'val_40.npy', 'val_60.npy']
        for name in names:
            download_url(f'{self.url}/{name}', self.dir)
           



    # onehot encoder
    def encode_onehot(self, labels):
        labels = labels.reshape(-1, 1)
        enc = OneHotEncoder()
        enc.fit(labels)
        labels_onehot = enc.transform(labels).toarray()
        return labels_onehot

    #preprocess for feature
    def preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features.todense()
    
    # normalize adjacency matrix
    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    
    # get processed data from ACM

    def data4HeCo (self, args):
        type_num = args.type_num
        #data process for meta path
        pa = np.genfromtxt(self.dir + "pa.txt")  #E:\\send_heco\\send_heco\\data\\acm\\
        ps = np.genfromtxt(self.dir + "ps.txt")
        label = np.load(self.dir + "labels.npy").astype('int32')
        feat_p = sp.load_npz(self.dir + "p_feat.npz")
        train_20 = np.load(self.dir + "train_20.npy")
        train_40 = np.load(self.dir + "train_40.npy")
        train_60 = np.load(self.dir + "train_60.npy")
        test_20 = np.load(self.dir + "test_20.npy")
        test_40 = np.load(self.dir + "test_40.npy")
        test_60 = np.load(self.dir + "test_60.npy")
        val_20 = np.load(self.dir + "val_20.npy")
        val_40 = np.load(self.dir + "val_40.npy")
        val_60 = np.load(self.dir + "val_60.npy")
    #print(relations)
        pa_row = []
        pa_col = []
        ps_row = []
        ps_col = []
        for rel_pa in pa:
            pa_row.append(rel_pa[0].astype(int))
            pa_col.append(rel_pa[1].astype(int))
        for rel_ps in ps:
            ps_row.append(rel_ps[0].astype(int))
            ps_col.append(rel_ps[1].astype(int))
        sc_pa = []
        sc_ps = []
        sc_pa.append(pa_row)
        sc_pa.append(pa_col)
        sc_ps.append(ps_row)
        sc_ps.append(ps_col)
        pa_ = sp.coo_matrix((np.ones(len(sc_pa[0])), (np.array(sc_pa[0]), np.array(sc_pa[1]))), shape=(np.max(sc_pa[0])+1, np.max(sc_pa[1])+1)).toarray()
        ps_ = sp.coo_matrix((np.ones(len(sc_ps[0])), (np.array(sc_ps[0]), np.array(sc_ps[1]))), shape=(np.max(sc_ps[0])+1, np.max(sc_ps[1])+1)).toarray()
        pap = np.matmul(pa_, pa_.T) > 0
        pap = sp.coo_matrix(pap)
        psp = np.matmul(ps_, ps_.T) > 0
        psp = sp.coo_matrix(psp)
        pos_num = 5
        p = 4019
        pap = pap / pap.sum(axis=-1).reshape(-1,1)
        psp = psp / psp.sum(axis=-1).reshape(-1,1)
        '''print(pap)
        print(psp)'''
        all = (pap + psp).A.astype("float32")

        #print(all) 
        #print(all_)
        #print(all_.max(),all_.min(),all_.mean())

        pos = np.zeros((p,p))
        k=0
        #print(len(all))
        for i in range(len(all)):
            one = all[i].nonzero()[0]
            if len(one) > pos_num:
                oo = np.argsort(-all[i, one])
                sele = one[oo[:pos_num]]
                pos[i, sele] = 1
                k+=1
            else:
                pos[i, one] = 1
        #print(one)
        pos = sp.coo_matrix(pos)
        pap = self.normalize_adj(pap)
        psp = self.normalize_adj(psp)
        pap = pap.todense().astype(np.float32)
        pap = tlx.convert_to_tensor(pap)
        psp = psp.todense().astype(np.float32)
        psp = tlx.convert_to_tensor(psp)
        pos = pos.todense().astype(np.float32)
        pos = tlx.convert_to_tensor(pos)


        #data process for labels 
        label = self.encode_onehot(label)
        label = tlx.convert_to_tensor(label)
        '''path = "E:\\send_heco\\send_heco\\data\\acm\\"
        label = np.load(path + "labels.npy").astype('int32')
        print(label[669])'''


        #data process for feature
        feat_a = sp.eye(type_num[1])  # 单位矩阵 7167*7167
        feat_s = sp.eye(type_num[2])
        feat_p = tlx.convert_to_tensor(self.preprocess_features(feat_p), 'float32')
        feat_a = tlx.convert_to_tensor(self.preprocess_features(feat_a), 'float32')
        feat_s = tlx.convert_to_tensor(self.preprocess_features(feat_s), 'float32')


        #data process for neibor
        p_n_a = {}
        p_n_s = {}
        for nei_pa in pa:
            if nei_pa[0] not in p_n_a:
                p_n_a[int(nei_pa[0])] = []
                p_n_a[int(nei_pa[0])].append(int(nei_pa[1]))
            else:
                p_n_a[int(nei_pa[0])].append(int(nei_pa[1]))
        keys =  sorted(p_n_a.keys())
        p_n_a = [p_n_a[i] for i in keys]
        p_n_a = [np.array(i) for i in p_n_a]
        for nei_ps in ps:
            if nei_ps[0] not in p_n_s:
                p_n_s[int(nei_ps[0])] = []
                p_n_s[int(nei_ps[0])].append(int(nei_ps[1]))
            else:
                p_n_s[int(nei_ps[0])].append(int(nei_ps[1]))
        keys =  sorted(p_n_s.keys())
        p_n_s = [p_n_s[i] for i in keys]
        p_n_s = [np.array(i) for i in p_n_s]
        nei_a = p_n_a
        nei_s = p_n_s
        nei_a = [tlx.convert_to_tensor(i, 'int64') for i in nei_a]
        nei_s = [tlx.convert_to_tensor(i, 'int64') for i in nei_s]

        #data process for train\test\val
        train = []
        test = []
        val = []
        train.append(train_20)
        train.append(train_40)
        train.append(train_60)
        test.append(test_20)
        test.append(test_40)
        test.append(test_60)
        val.append(val_20)
        val.append(val_40)
        val.append(val_60)
        train = [tlx.convert_to_tensor(i, 'int64') for i in train]
        val = [tlx.convert_to_tensor(i, 'int64') for i in val]
        test = [tlx.convert_to_tensor(i, 'int64') for i in test]
        return [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], pos, label, train, val, test