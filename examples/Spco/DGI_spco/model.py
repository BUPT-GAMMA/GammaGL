import os

import tensorlayerx as tlx
# from gammagl.layers.conv import GCNConv
from MyGCN import GCNConv
import math
import numpy as np


class LogReg(tlx.nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = tlx.nn.Linear(in_features=hid_dim, out_features=out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class Discriminator(tlx.nn.Module):
    def __init__(self, n_hid):
        super(Discriminator, self).__init__()

        init = tlx.nn.initializers.RandomUniform(-1.0 / math.sqrt(n_hid), 1.0 / math.sqrt(n_hid))
        self.fc = tlx.nn.Linear(in_features=n_hid, out_features=n_hid, W_init=init)

    def forward(self, feat, summary):
        feat = tlx.matmul(self.fc(feat), tlx.expand_dims(summary, -1))

        return feat


class GCN(tlx.nn.Module):
    def __init__(self, in_ft, out_ft, act, add_bias=True):
        super(GCN, self).__init__()
        self.conv = GCNConv(in_ft, out_ft, add_bias=add_bias)
        self.act = act

    def forward(self, feat, edge_index, edge_weight, num_nodes, cor=False):
        if cor == False:
            x = self.conv(feat, edge_index, edge_weight, num_nodes)
        else:
            perm = np.random.permutation(num_nodes)
            feat = tlx.gather(feat, tlx.convert_to_tensor(perm, dtype=tlx.int64))
            x = self.conv(feat, edge_index, edge_weight)
        return self.act(x)


class DGIModel(tlx.nn.Module):
    def __init__(self, in_feat, hid_feat, act):
        super(DGIModel, self).__init__()
        self.gcn = GCN(in_feat, hid_feat, act)
        self.disc = Discriminator(hid_feat)
        self.loss = tlx.losses.sigmoid_cross_entropy

    def get_embedding(self, feat, edge_index, edge_weight):
        out = self.gcn(feat, edge_index, edge_weight, feat.shape[0])
        return out.detach()

    def forward(self, feat_1, edge_index_1, edge_weight_1, feat_2, edge_index_2, edge_weight_2):
        pos = self.gcn(feat_1, edge_index_1, edge_weight_1, feat_1.shape[0])
        neg = self.gcn(feat_2, edge_index_2, edge_weight_2, feat_2.shape[0], True)

        summary = tlx.sigmoid(tlx.reduce_mean(pos, axis=0))

        pos = self.disc(pos, summary)
        neg = self.disc(neg, summary)

        pos_loss = self.loss(pos, tlx.ones_like(pos))
        neg_loss = self.loss(neg, tlx.zeros_like(neg))

        return pos_loss + neg_loss
