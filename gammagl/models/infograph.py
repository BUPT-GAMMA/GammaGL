"""
References
----------
Papers:  https://arxiv.org/pdf/1908.01000
Author's code: https://github.com/fanyun-sun/InfoGraph
"""

import tensorlayerx as tlx
import tensorlayerx.nn as nn

from gammagl.utils.tu_utils import local_global_loss_
from gammagl.layers.conv import GINConv
from gammagl.layers.pool.glob import global_sum_pool


class FF(nn.Module):
    r"""
        Graph Convolution Encoder Architect

        Parameters
        ----------
        in_feat: int
            input feature dimension
        hid_feat: int
            hidden dimension

    """
    def __init__(self, in_feat, hid_feat):
        super(FF, self).__init__()
        self.block = nn.Sequential(
            tlx.nn.Linear(in_features=in_feat, W_init=tlx.initializers.xavier_uniform(), out_features=hid_feat),
            nn.ReLU(),
            tlx.nn.Linear(in_features=in_feat, W_init=tlx.initializers.xavier_uniform(), out_features=hid_feat),
            nn.ReLU(),
            tlx.nn.Linear(in_features=in_feat, W_init=tlx.initializers.xavier_uniform(), out_features=hid_feat),
            nn.ReLU()
        )
        self.jump_con = tlx.nn.Linear(in_features=in_feat, W_init=tlx.initializers.xavier_uniform(),
                                      out_features=hid_feat)

    def forward(self, x):
        return self.block(x) + self.jump_con(x)


class PriorDiscriminator(nn.Module):
    def __init__(self, in_feat):
        super().__init__()
        self.l0 = nn.Linear(out_features=in_feat, W_init=tlx.initializers.xavier_uniform(), in_features=in_feat)
        self.l1 = nn.Linear(out_features=in_feat, W_init=tlx.initializers.xavier_uniform(), in_features=in_feat)
        self.l2 = nn.Linear(out_features=1, W_init=tlx.initializers.xavier_uniform(), in_features=in_feat)

    def forward(self, x):
        h = tlx.ReLU()(self.l0(x))
        h = tlx.ReLU()(self.l1(h))
        return tlx.sigmoid(self.l2(h))


class GINEncoder(nn.Module):
    """
        GIN encoder

        Parameters
        ----------
        num_feature: int
            input feature dimension
        hid_feat: int
            hidden dimension
        num_gc_layers: int
            number of layers
    """
    def __init__(self, num_feature, out_feat, num_gc_layers):
        super(GINEncoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        convs_list = []
        bns_list = []
        for i in range(num_gc_layers):
            if i == 0:
                n_in = num_feature
            else:
                n_in = out_feat
            n_out = out_feat
            block = nn.Sequential(
                nn.Linear(out_features=n_out, W_init=tlx.initializers.xavier_uniform(), in_features=n_in),
                nn.ReLU(),
                nn.Linear(out_features=n_out, W_init=tlx.initializers.xavier_uniform(), in_features=n_out)
                )
            conv = GINConv(nn=block)
            bn = nn.BatchNorm1d(num_features=out_feat)
            convs_list.append(conv)
            bns_list.append(bn)
            self.convs = nn.Sequential(convs_list)
            self.bns = nn.Sequential(bns_list)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, batch):
        if x is None:
            x = tlx.ones((batch.shape[0], 1), dtype=tlx.float32)

        xs = []
        for i in range(self.num_gc_layers):
            x = self.act(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        sum_pool = [global_sum_pool(x, batch) for x in xs]
        local_emb = tlx.concat(xs, 1)  # patch-level embedding
        global_emb = tlx.concat(sum_pool, 1)  # graph-level embedding

        return global_emb, local_emb


class InfoGraph(nn.Module):
    """infograph model for unsupervised setting

        Parameters
        ----------
        num_feature: int
            input feature dimension.
        hid_feat: int
            hidden dimension.
        num_gc_layers: int
            number of layers.
        prior: bool
            whether to use prior discriminator.
        gamma: float
            tunable hyper-parameter.
    """

    def __init__(self, num_feature, hid_feat, num_gc_layers, prior, gamma=.1):
        super(InfoGraph, self).__init__()
        embedding_dim = num_gc_layers * hid_feat
        self.prior = prior
        self.gamma = gamma
        self.encoder = GINEncoder(num_feature, hid_feat, num_gc_layers)

        self.local_d = FF(embedding_dim, embedding_dim)  # local discriminator (node-level)
        self.global_d = FF(embedding_dim, embedding_dim)  # global discriminator (graph-level)
        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)

    def get_embedding(self, dataloader):
        """
        Evaluation the learned embeddings

        Parameters
        ----------
        dataloader:
            dataloader of dataset.

        Returns
        -------
        res: tensor
            concat of x.
        y: tensor
            concat of y.
        """
        res = []
        y = []
        for data in dataloader:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            if x is None:
                x = tlx.ones((batch.shape[0], 1), dtype=tlx.float32)
            x, _ = self.encoder(x, edge_index, batch)
            res.append(x)
            y.append(data.y)
        res = tlx.concat(res, axis=0)
        y = tlx.concat(y, axis=0)
        return tlx.convert_to_tensor(res), tlx.convert_to_tensor(y)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = tlx.ones((batch.shape[0], 1), dtype=tlx.float32)
        global_emb, local_emb = self.encoder(x, edge_index, batch)
        global_h = self.global_d(global_emb)
        local_h = self.local_d(local_emb)
        loss = local_global_loss_(local_h, global_h, batch)
        if self.prior:
            prior = tlx.random_uniform(len(global_h))
            term_a = tlx.reduce_mean(tlx.log(self.prior_d(prior)))
            term_b = tlx.reduce_mean(tlx.log(1.0 - self.prior_d(global_h)))
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        return loss + PRIOR
