
import scipy.sparse as sp
import numpy as np
import os
from sklearn.cluster import KMeans
import tensorlayerx as tlx
from tensorlayerx import nn as nn
from examples.dyfss.utils.compat import log_softmax, gather
from examples.dyfss.utils.layers import GraphConvolution

class Clu(nn.Module):
    def __init__(self, data, processed_data, encoder, nhid1, nhid2, dropout, **kwargs):
        super(Clu, self).__init__()
        self.args = kwargs['args']
        self.data = data
        self.processed_data = processed_data
        self.ncluster = 10
        self.pseudo_labels = self.get_label(self.ncluster)
        self.pseudo_labels = tlx.convert_to_tensor(self.pseudo_labels, dtype=tlx.int64)
        self.disc1 = nn.Linear(in_features=nhid1, out_features=self.ncluster)
        self.sampled_indices = (self.pseudo_labels >= 0)
        self.gcn = encoder
        self.gcn2 = GraphConvolution(nhid2, nhid2, dropout, act=lambda x: x)
        self.disc2 = nn.Linear(in_features=nhid2, out_features=self.ncluster)

    def make_loss_stage_two(self, encoder_features, adj_norm):
        embeddings = self.gcn2_forward(encoder_features, adj_norm)
        embeddings = self.disc2(embeddings)
        output = log_softmax(embeddings, dim=1)
        valid_indices = tlx.convert_to_numpy(self.sampled_indices)
        valid_indices = np.where(valid_indices)[0]
        valid_indices = tlx.convert_to_tensor(valid_indices, dtype=tlx.int64)
        valid_output = gather(output, valid_indices, axis=0)
        valid_labels = gather(self.pseudo_labels, valid_indices, axis=0)
        loss = tlx.losses.softmax_cross_entropy_with_logits(valid_output, valid_labels)
        return loss

    def make_loss_stage_one(self, embeddings):
        embeddings = self.disc1(embeddings)
        output = log_softmax(embeddings, dim=1)
        valid_indices = tlx.convert_to_numpy(self.sampled_indices)
        valid_indices = np.where(valid_indices)[0]
        valid_indices = tlx.convert_to_tensor(valid_indices, dtype=tlx.int64)
        valid_output = gather(output, valid_indices, axis=0)
        valid_labels = gather(self.pseudo_labels, valid_indices, axis=0)
        
        loss = tlx.losses.softmax_cross_entropy_with_logits(valid_output, valid_labels)
        return loss

    def gcn2_forward(self, input, adj):
        self.train()
        embeddings = self.gcn2(input, adj)
        return embeddings

    def get_label(self, ncluster):
        cluster_file = './saved/' + self.args.dataset + '_cluster_%s.npy' % ncluster
        if not os.path.exists(cluster_file):
            print('perform clustering with KMeans...')
            features = tlx.convert_to_numpy(self.data.features) if not isinstance(self.data.features, np.ndarray) else self.data.features
            kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(features)
            cluster_labels = kmeans.labels_
            return cluster_labels
        else:
            cluster_labels = np.load(cluster_file)
            return cluster_labels
