"""
Deep Graph Infomax in DGL

References
----------
Papers: https://arxiv.org/abs/1809.10341
Author's code: https://github.com/PetarV-/DGI
"""

import tensorlayerx as tlx
import math

from gammagl.layers.conv import GCNConv


class Discriminator(tlx.nn.Module):
    def __init__(self, n_hid):
        super(Discriminator, self).__init__()

        init = tlx.initializers.RandomUniform(-1.0 / math.sqrt(n_hid), 1.0 / math.sqrt(n_hid))
        self.ww = tlx.nn.Linear(in_features=n_hid, out_features=n_hid, W_init=init)

    def forward(self, feat, summary):
        feat = tlx.matmul(feat, tlx.matmul(self.ww.W, tlx.expand_dims(summary, -1)))
        return feat


class GCN(tlx.nn.Module):
    def __init__(self, in_ft, out_ft, act, add_bias=True):
        super(GCN, self).__init__()
        self.conv = GCNConv(in_ft, out_ft, add_bias=add_bias)
        self.act = act

    def forward(self, feat, edge_index, edge_weight, num_nodes):
        x = self.conv(feat, edge_index, edge_weight, num_nodes)
        return self.act(x)


class DGIModel(tlx.nn.Module):
    def __init__(self, in_feat, hid_feat, act):
        super(DGIModel, self).__init__()
        self.gcn = GCN(in_feat, hid_feat, act)
        self.disc = Discriminator(hid_feat)
        self.loss = tlx.losses.sigmoid_cross_entropy

    def forward(self, feat1, feat2, edge_index, edge_weight, num_nodes):
        pos = self.gcn(feat1, edge_index, edge_weight, num_nodes)
        neg = self.gcn(feat2, edge_index, edge_weight, num_nodes)

        summary = tlx.sigmoid(tlx.reduce_mean(pos, axis=0))

        pos = self.disc(pos, summary)
        neg = self.disc(neg, summary)

        pos_loss = self.loss(pos, tlx.ones(pos.shape))
        neg_loss = self.loss(neg, tlx.zeros(neg.shape))

        return pos_loss + neg_loss

import numpy as np
import scipy.sparse as sp
def calc(edge, num_node):
    weight = np.ones(edge.shape[1])
    sparse_adj = sp.coo_matrix((weight, (edge[0], edge[1])), shape=(num_node, num_node))
    A = (sparse_adj + sp.eye(num_node)).tocoo()
    col, row, weight = A.col, A.row, A.data
    deg = np.array(A.sum(1))
    deg_inv_sqrt = np.power(deg, -0.5).flatten()
    return col, row, np.array(deg_inv_sqrt[row] * weight * deg_inv_sqrt[col], dtype=np.float32)
