"""
Deep Graph Infomax in DGL

References
----------
Papers: https://arxiv.org/abs/1809.10341
Author's code: https://github.com/PetarV-/DGI
"""
import numpy as np
import tensorlayerx.nn as nn
import tensorlayerx as tlx
import math

from gammagl.layers.conv import GCNConv


class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden, activation):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_feats, n_hidden)
        self.act = activation

    def forward(self, features, edge_index, edge_weight, num_nodes, corrupt=False):
        if corrupt:
            perm = np.random.permutation(num_nodes)
            features = tlx.gather(features, tlx.convert_to_tensor(perm, dtype=tlx.int64))
        features = self.conv(features, edge_index, edge_weight, num_nodes)
        return self.act(features)


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        init = tlx.nn.initializers.random_uniform(- 1.0 / math.sqrt(n_hidden), 1.0 / math.sqrt(n_hidden))
        self.fc = tlx.nn.Linear(in_features=n_hidden, out_features=n_hidden, W_init=init)


    def forward(self, features, summary):
        features = tlx.matmul(self.fc(features), tlx.expand_dims(summary, -1))
        return features


class DGIModel(nn.Module):
    def __init__(self, in_feats, n_hidden, activation):
        super(DGIModel, self).__init__()
        self.encoder = Encoder(in_feats, n_hidden, activation)
        self.discriminator = Discriminator(n_hidden)
        self.loss = tlx.losses.sigmoid_cross_entropy

    def forward(self, features, edge_index, edge_weight, num_nodes):
        positive = self.encoder(features, edge_index, edge_weight, num_nodes, corrupt=False)
        negative = self.encoder(features, edge_index, edge_weight, num_nodes, corrupt=True)
        summary = tlx.sigmoid(tlx.reduce_mean(positive, 0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, tlx.ones_like(positive))
        l2 = self.loss(negative, tlx.zeros_like(negative))

        return l1 + l2


