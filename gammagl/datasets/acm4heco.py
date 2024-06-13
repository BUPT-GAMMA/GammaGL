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
from gammagl.data import HeteroGraph, InMemoryDataset, extract_zip
from sklearn.preprocessing import OneHotEncoder
import requests
import io
import zipfile
from gammagl.data import download_url

class ACM4HeCo(InMemoryDataset):
    r"""The heterogeneous ACM dataset from the `"Self-supervised Heterogeneous Graph Neural Network with
    Co-contrastive Learning"
    <https://arxiv.org/abs/2105.09111>`_ paper, consisting of nodes from
    type :obj:`"paper"`, :obj:`"author"` and :obj:`"subject"`.
    ACM is a heterogeneous graph containing three types of entities - paper
    (4019 nodes), author (7167 nodes), and subject (60 nodes).

    Parameters
    ----------
    root: str, optional
        Root directory where the dataset should be saved.
    transform: callable, optional
        A function/transform that takes in an
        :obj:`gammagl.data.HeteroGraph` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform: callable, optional
        A function/transform that takes in
        an :obj:`gammagl.data.HeteroGraph` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

    """
    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None,
                 force_reload: bool = False):
        self.url = 'https://github.com/liun-online/HeCo/raw/main/data/acm'
        self.name = 'ACM4HeCo'
        super().__init__(root, transform, pre_transform, pre_filter, force_reload = force_reload)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['pa.txt', 'ps.txt', 'labels.npy', 'p_feat.npz', 'train_20.npy', 
                'train_40.npy', 'train_60.npy', 'test_20.npy', 'test_40.npy', 'test_60.npy',
                  'val_20.npy', 'val_40.npy', 'val_60.npy']
    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + '_data.pt'    
    def download(self):
        for i in range(0, len(self.raw_file_names)):
           download_url(f'{self.url}/{self.raw_file_names[i]}', self.raw_dir)           
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

    def process (self):
        data = HeteroGraph()
        type_num = [4019, 7167, 60]
        
        #data process for meta path
        pa = np.genfromtxt(osp.join(self.raw_dir,'pa.txt'))  #E:\\send_heco\\send_heco\\data\\acm\\
        ps = np.genfromtxt(osp.join(self.raw_dir,'ps.txt'))
        label = np.load(osp.join(self.raw_dir,'labels.npy')).astype('int32')
        feat_p = sp.load_npz(osp.join(self.raw_dir,'p_feat.npz'))
        train_20 = np.load(osp.join(self.raw_dir,'train_20.npy'))
        train_40 = np.load(osp.join(self.raw_dir,'train_40.npy'))
        train_60 = np.load(osp.join(self.raw_dir,'train_60.npy'))
        test_20 = np.load(osp.join(self.raw_dir,'test_20.npy'))
        test_40 = np.load(osp.join(self.raw_dir,'test_40.npy'))
        test_60 = np.load(osp.join(self.raw_dir,'test_60.npy'))
        val_20 = np.load(osp.join(self.raw_dir,'val_20.npy'))
        val_40 = np.load(osp.join(self.raw_dir,'val_20.npy'))
        val_60 = np.load(osp.join(self.raw_dir,'val_20.npy'))
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

        #data process for feature
        feat_a = sp.eye(type_num[1])  # 单位矩阵 7167*7167
        feat_s = sp.eye(type_num[2])
        feat_p = tlx.convert_to_tensor(self.preprocess_features(feat_p), 'float32')
        feat_a = tlx.convert_to_tensor(self.preprocess_features(feat_a), 'float32')
        feat_s = tlx.convert_to_tensor(self.preprocess_features(feat_s), 'float32')

        #data process for edge_index
        p_list = []
        a_list = []
        row_num = np.size(pa, 0)
        for i in range(0, row_num):
            p_list.append(pa[i][0])
            a_list.append(pa[i][1])
        edge_idx_pa = sp.coo_matrix((np.ones(len(p_list)), (np.array(p_list), np.array(a_list))), shape=(4019, 7167)).toarray()

        p_list = []
        s_list = []
        row_num = np.size(ps, 0)
        for i in range(0, row_num):
            p_list.append(ps[i][0])
            s_list.append(ps[i][1])
        edge_idx_ps = sp.coo_matrix((np.ones(len(p_list)), (np.array(p_list), np.array(s_list))), shape=(4019, 60)).toarray()

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
        data['edge_index_dict'] = {('paper', 'to', 'author'): edge_idx_pa, ('paper', 'to', 'subject'): edge_idx_ps}
        data['paper'].x = feat_p
        data['author'].x = feat_a
        data['subject'].x = feat_s
        data['metapath'] = [pap, psp]
        data['pos_set_for_contrast'] = pos
        data['paper'].y = label
        data['train'] = train
        data['val'] = val
        data['test'] = test
        data['nei_num'] = 2
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self.save_data(self.collate([data]), self.processed_paths[0])

