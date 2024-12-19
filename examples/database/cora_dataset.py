import os
import os.path as osp
import os
import tensorlayerx as tlx

import numpy as np
import scipy.sparse as sp
from gammagl.data import extract_zip, download_url, InMemoryDataset, Graph
from gammagl.sparse.coalesce import coalesce
import pickle
from sklearn.model_selection import train_test_split
import torch
class Cora(InMemoryDataset):

    def __init__(self, db_dataset):
        self.db_dataset = db_dataset
        self.graph = self.processCora()

    def processCora(self):
        data = self.db_dataset
        x = tlx.convert_to_tensor(data['X_dict']['node'], dtype=tlx.float32)
        y = tlx.convert_to_tensor(data['Y_dict']['node'], dtype=tlx.int64)

        edge = data['edge_index_dict']['edge']
        data = Graph(edge_index=edge, x=x, y=y)


        #split dataset
        X_num = self.db_dataset['X_dict']['node'].shape[0] 
        X_ids = np.arange(X_num)
        X_train_val, X_test = train_test_split(X_ids, test_size=0.2, random_state=42)
        X_train, X_val = train_test_split(X_train_val, test_size=0.25, random_state=42)
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
        train_idx = tlx.convert_to_tensor(idx_train)
        test_idx = tlx.convert_to_tensor(idx_val)
        val_idx = tlx.convert_to_tensor(idx_test)
        data.train_idx = train_idx 
        data.test_idx= test_idx
        data.val_idx = val_idx
        return data
