
from .discriminator import Discriminator
from .gnn_encoder import AvgReadout
import numpy as np
from deeprobust.graph import utils

import tensorlayerx as tlx
from tensorlayerx import nn as nn

from examples.dyfss.utils.layers import GraphConvolution


class DGI(nn.Module):

    def __init__(self, data, processed_data, encoder, nhid1, nhid2, dropout, device, **kwargs):
        super(DGI, self).__init__()
        self.args = kwargs['args']
        self.data = data
        self.processed_data = processed_data
        self.device = device
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc1 = Discriminator(nhid1)
        self.b_xent = tlx.losses.sigmoid_cross_entropy
        self.encoder = encoder
        self.gcn2 = GraphConvolution(nhid2, nhid2, dropout, act=lambda x: x)
        self.disc2 = Discriminator(nhid2)
        pseudo_labels = self.get_label()
        self.pseudo_labels = pseudo_labels
        self.num_nodes = data.adj.shape[0]

    def get_label(self):
        nb_nodes = self.processed_data.features.shape[0]
        lbl_1 = tlx.ones((nb_nodes,))
        lbl_2 = tlx.zeros((nb_nodes,))
        lbl = tlx.concat([lbl_1, lbl_2], axis=0)
        return lbl

    def make_loss_stage_two(self, encoder_features, adj_norm):
        features = self.processed_data.features
        adj = self.processed_data.adj_norm
        nb_nodes = features.shape[0]

        self.train()
        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[idx, :]
        embeddings = self.gcn2_forward(encoder_features, adj_norm)
        logits = self.forward2(features, shuf_fts, adj, None, None, None, None, embedding=embeddings)
        loss = self.b_xent(logits, self.pseudo_labels)
        return loss

    def gcn2_forward(self, input, adj):
        embeddings = self.gcn2(input, adj)
        return embeddings

    def forward2(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2, embedding=None):
        if embedding is None:
            _, _, _, _, h_1 = self.encoder(seq1, adj)
        else:
            h_1 = embedding
        c = self.read(h_1, msk)
        c = self.sigm(c)
        _, _, _, _, h_2 = self.encoder(seq2, adj)
        h_2 = self.gcn2_forward(h_2, adj)
        ret = self.disc2(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret

    def make_loss_stage_one(self, x):
        features = self.processed_data.features
        adj = self.processed_data.adj_norm
        nb_nodes = features.shape[0]

        self.train()
        idx = np.random.permutation(nb_nodes)

        shuf_fts = features[idx, :]

        logits = self.forward1(features, shuf_fts, adj, None, None, None, None, embedding=x)
        loss = self.b_xent(logits, self.pseudo_labels)
        # print('Loss:', loss.item())
        return loss

    def forward1(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2, embedding=None):
        if embedding is None:
            h_1, _, _, _, _ = self.encoder(seq1, adj)
        else:
            h_1 = embedding
        c = self.read(h_1, msk)
        c = self.sigm(c)
        h_2, _, _, _, _ = self.encoder(seq2, adj)
        ret = self.disc1(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret




class DGISample(nn.Module):  

    def __init__(self, data, processed_data, encoder, nhid1, nhid2, dropout, device, **kwargs):
        super(DGISample, self).__init__()
        self.args = kwargs['args']
        self.data = data
        self.processed_data = processed_data
        self.device = device
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc1 = Discriminator(nhid1)
        self.b_xent = tlx.losses.sigmoid_cross_entropy
        self.encoder = encoder
        self.gcn2 = GraphConvolution(nhid2, nhid2, dropout, act=lambda x: x)
        self.disc2 = Discriminator(nhid2)

        if kwargs['args'].dataset in ['reddit', 'arxiv']:
            self.sample_size = 2000
        else:
            self.sample_size = 2000
        self.pseudo_labels = self.get_label()

        self.b_xent = tlx.losses.sigmoid_cross_entropy        
        self.num_nodes = data.adj.shape[0]

    def get_label(self):
   
        lbl_1 = tlx.ones((self.sample_size,))
        lbl_2 = tlx.zeros((self.sample_size,))
        lbl = tlx.concat([lbl_1, lbl_2], axis=0)
        return lbl

    def gcn2_forward(self, input, adj):

        features = self.gcn2(input, adj)
        return features

    def make_loss_stage_two(self, encoder_features, adj_norm):
        features = self.processed_data.features
        adj = self.processed_data.adj_norm
        nb_nodes = features.shape[0]

        self.train()
        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[idx, :]

        embeddings = self.gcn2_forward(encoder_features, adj_norm)

        logits = self.forward2(features, shuf_fts, adj, None, None, None, None, embeddings)
        loss = self.b_xent(logits, self.pseudo_labels)
        return loss

    def forward2(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2, embedding=None):
        idx = np.random.default_rng(self.args.seed).choice(self.num_nodes,
                self.sample_size, replace=False)
        if embedding is None:
            _, _, _, _, h_1 = self.encoder(seq1, adj)[idx]
        else:
            h_1 = embedding[idx]
        c = self.read(h_1, msk)
        c = self.sigm(c)

        _, _, _, _, h_2 = self.encoder(seq2, adj)
        h_2 = self.gcn2_forward(h_2, adj)[idx]

        ret = self.disc2(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret

    def make_loss_stage_one(self, x):
        features = self.processed_data.features
        adj = self.processed_data.adj_norm
        nb_nodes = features.shape[0]

        self.train()
        idx = np.random.permutation(nb_nodes)

        shuf_fts = features[idx, :]

        logits = self.forward1(features, shuf_fts, adj, None, None, None, None, embedding=x)
        loss = self.b_xent(logits, self.pseudo_labels)

        return loss

    def forward1(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2, embedding=None):
        idx = np.random.default_rng(self.args.seed).choice(self.num_nodes,
                            self.sample_size, replace=False)
        if embedding is None:
            h_1, _, _, _, _ = self.encoder(seq1, adj)
            h_1 = h_1[idx]
        else:
            h_1 = embedding[idx]
        c = self.read(h_1, msk)
        c = self.sigm(c)
        h_2, _, _, _, _ = self.encoder(seq2, adj)
        h_2 = h_2[idx]
        ret = self.disc1(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret



