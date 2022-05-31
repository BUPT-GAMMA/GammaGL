import math

import tensorlayerx as tlx
from gammagl.layers.conv import GCNConv
from gammagl.utils import add_self_loops, calc_gcn_norm

from gammagl.utils.tu_utils import local_global_loss_

# ======================= Graph Model================================

class MLP(tlx.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fcs = tlx.nn.Sequential(
            tlx.nn.Linear(in_features=in_dim, out_features=out_dim),
            tlx.nn.PRelu(),
            tlx.nn.Linear(in_features=out_dim, out_features=out_dim),
            tlx.nn.PRelu(),
            tlx.nn.Linear(in_features=out_dim, out_features=out_dim),
            tlx.nn.PRelu(),
        )
        self.linear_shortcut = tlx.nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, x):
        return self.fcs(x) + self.linear_shortcut(x)


class GCN_graph(tlx.nn.Module):
    def __init__(self, in_feat, out_feat, num_layers):
        super(GCN_graph, self).__init__()
        self.num_layers = num_layers

        self.act = tlx.nn.PRelu()

        self.convs = tlx.nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_feat, out_feat, add_bias=False))
        else:
            self.convs.append(GCNConv(in_feat, out_feat, add_bias=False))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(out_feat, out_feat, add_bias=False))


    def forward(self, feat, edge_index, edge_weight, num_nodes, ptr):
        h = self.convs[0](feat, edge_index, edge_weight, num_nodes)
        h = self.act(h)
        hg = self.divide_graph(h, ptr)

        for i in range(1, self.num_layers):
            h = self.convs[0](feat, edge_index, edge_weight, num_nodes)
            hg = tlx.concat((hg, self.divide_graph(h, ptr)), -1)
        return h, hg
    def divide_graph(self, hg, ptr):
        hs = []
        for i in range(len(ptr) - 1) :
            hs.append(tlx.expand_dims(tlx.reduce_sum(tlx.gather(hg, tlx.arange(ptr[i], ptr[i+1])), axis=0), axis=0))
        return tlx.concat(hs, axis=0)


class MVGRL_Graph(tlx.nn.Module):
    def __init__(self, in_feat, out_feat, num_layers):
        super(MVGRL_Graph, self).__init__()
        self.local_mlp = MLP(out_feat, out_feat)
        self.global_mlp = MLP(num_layers * out_feat, out_feat)
        self.encoder1 = GCN_graph(in_feat, out_feat, num_layers)
        self.encoder2 = GCN_graph(in_feat, out_feat, num_layers)


    def get_embedding(self, edge_index, diff_edge, diff_weight, feat, ptr):
        # calculate node embeddings and graph embeddings
        if tlx.BACKEND == 'torch':
            edge_index = edge_index.cpu()
        edge_index, _ = add_self_loops(edge_index)
        norm_weight = calc_gcn_norm(edge_index, feat.shape[0])
        if tlx.BACKEND == 'torch':
            edge_index = edge_index.to(feat.device)
            norm_weight = norm_weight.to(feat.device)
        local_v1, global_v1 = self.encoder1(feat, edge_index, norm_weight, feat.shape[0], ptr)
        local_v2, global_v2 = self.encoder2(feat, diff_edge, diff_weight, feat.shape[0], ptr)

        global_v1 = self.global_mlp(global_v1)
        global_v2 = self.global_mlp(global_v2)
        if tlx.BACKEND != 'tensorflow':
            return (global_v1 + global_v2).detach()
        else:
            return global_v1 + global_v2

    def forward(self, edge_index, diff_edge, diff_weight, feat, ptr, batch):
        # calculate node embeddings and graph embeddings
        if tlx.BACKEND == 'torch':
            edge_index = edge_index.cpu()
        edge_index, _ = add_self_loops(edge_index)
        norm_weight = calc_gcn_norm(edge_index, feat.shape[0])
        if tlx.BACKEND == 'torch':
            edge_index = edge_index.to(feat.device)
            norm_weight = norm_weight.to(feat.device)
        local_v1, global_v1 = self.encoder1(feat, edge_index, norm_weight, feat.shape[0], ptr)
        local_v2, global_v2 = self.encoder2(feat, diff_edge, diff_weight, feat.shape[0], ptr)

        local_v1 = self.local_mlp(local_v1)
        local_v2 = self.local_mlp(local_v2)

        global_v1 = self.global_mlp(global_v1)
        global_v2 = self.global_mlp(global_v2)

        # calculate loss
        loss1 = local_global_loss_(local_v1, global_v2, batch)
        loss2 = local_global_loss_(local_v2, global_v1, batch)

        loss = loss1 + loss2

        return loss

# ========================= Node Model==============================

class GCNConv_Dense(tlx.nn.Module):
    def __init__(self, in_dim, out_dim, add_bias=True):
        super(GCNConv_Dense, self).__init__()
        init = tlx.nn.initializers.XavierUniform()
        self.fc = tlx.nn.Linear(in_features=in_dim, out_features=out_dim, W_init=init, b_init=None)

        if add_bias is True:
            initor = tlx.initializers.zeros()
            self.bias = self._get_weights("bias", shape=(1, out_dim), init=initor)

    def forward(self, adj, feat):
        """
        :param adj: dense adj
        :param feat: node's feature
        :return: AHW
        """
        feat = self.fc(feat)
        feat = tlx.matmul(adj, feat)
        if self.bias is not None:
            feat += self.bias
        return feat


class LogReg(tlx.nn.Module):
    def __init__(self, hid_feat, n_classes):
        super(LogReg, self).__init__()

        self.fc = tlx.nn.Linear(in_features=hid_feat, out_features=n_classes)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class Discriminator(tlx.nn.Module):
    def __init__(self, n_hid):
        super(Discriminator, self).__init__()
        init = tlx.initializers.RandomUniform(-1.0 / math.sqrt(n_hid), 1.0 / math.sqrt(n_hid))

        self.fn = tlx.nn.Linear(in_features=n_hid, out_features=n_hid, W_init=init)

    def forward(self, h1, h2, h3, h4, c1, c2):
        # positive
        sc1 = tlx.reduce_sum(self.fn(h2) * c1, axis=1)
        sc2 = tlx.reduce_sum(self.fn(h1) * c2, axis=1)

        # negative
        sc3 = tlx.reduce_sum(self.fn(h4) * c1, axis=1)
        sc4 = tlx.reduce_sum(self.fn(h3) * c2, axis=1)

        return tlx.concat((sc1, sc2, sc3, sc4), axis=0)



class MVGRL(tlx.nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MVGRL, self).__init__()

        self.act = tlx.nn.PRelu()
        self.encoder1 = GCNConv(in_dim, out_dim, add_bias=True)
        self.encoder2 = GCNConv_Dense(in_dim, out_dim, add_bias=True)
        # self.pooling = AvgPooling()


        self.disc = Discriminator(out_dim)
        self.act_fn = tlx.nn.Sigmoid()
        self.loss = tlx.losses.sigmoid_cross_entropy

    def get_embedding(self, edge_index, diff_adj, feat):

        edge_index, _ = add_self_loops(edge_index)
        norm_weight = calc_gcn_norm(edge_index, feat.shape[0])
        h1 = self.encoder1(feat, edge_index, norm_weight, feat.shape[0])
        h2 = self.encoder2(diff_adj, feat)
        h1 = self.act(h1)
        h2 = self.act(h2)

        return h1 + h2
    def forward(self, edge_index, diff_adj, feat, shuf_feat):

        if tlx.BACKEND == 'torch':
            edge_index = edge_index.cpu()
        edge_index, _ = add_self_loops(edge_index)
        edge_weight = calc_gcn_norm(edge_index, feat.shape[0])
        if tlx.BACKEND == 'torch':
            edge_index = edge_index.to(feat.device)
            edge_weight = edge_weight.to(feat.device)

        h1 = self.act(self.encoder1(feat, edge_index, edge_weight, feat.shape[0]))
        h2 = self.act(self.encoder2(diff_adj, feat))

        h3 = self.act(self.encoder1(shuf_feat, edge_index, edge_weight, feat.shape[0]))
        h4 = self.act(self.encoder2(diff_adj, shuf_feat))

        c1 = self.act_fn(tlx.reduce_mean(h1, axis=0))
        c2 = self.act_fn(tlx.reduce_mean(h2, axis=0))
        out = self.disc(h1, h2, h3, h4, c1, c2)
        lbl1 = tlx.ones((feat.shape[0] * 2,))
        if tlx.BACKEND == 'torch':
            lbl1 = lbl1.to(out.device)
        lbl2 = tlx.zeros_like(lbl1)
        lbl = tlx.concat((lbl1, lbl2), axis=0)
        return self.loss(out, lbl)



