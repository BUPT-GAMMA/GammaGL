import math

import tensorlayerx as tlx
from gammagl.layers.conv import GCNConv


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
        edge_index, norm_weight = calc(edge_index, feat.shape[0])
        local_v1, global_v1 = self.encoder1(feat, edge_index, norm_weight, feat.shape[0], ptr)
        local_v2, global_v2 = self.encoder2(feat, diff_edge, diff_weight, feat.shape[0], ptr)

        global_v1 = self.global_mlp(global_v1)
        global_v2 = self.global_mlp(global_v2)

        return global_v1 + global_v2

    def forward(self, edge_index, diff_edge, diff_weight, feat, ptr, batch):
        # calculate node embeddings and graph embeddings
        edge_index, norm_weight = calc(edge_index, feat.shape[0])
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
        self.fc = tlx.nn.Linear(in_features=in_dim, out_features=out_dim)

        if add_bias is True:
            initor = tlx.initializers.truncated_normal()
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
        # sc1 = tlx.reduce_sum(tlx.matmul(h2, self.fn.W) * c1, axis=1)
        # sc2 = tlx.reduce_sum(tlx.matmul(h1, self.fn.W) * c2, axis=1)
        sc1 = tlx.reduce_sum(self.fn(h2) * c1, axis=1)
        sc2 = tlx.reduce_sum(self.fn(h1) * c2, axis=1)

        # negative
        # sc3 = tlx.reduce_sum(tlx.matmul(h4, self.fn.W) * c1, axis=1)
        # sc4 = tlx.reduce_sum(tlx.matmul(h3, self.fn.W) * c2, axis=1)
        sc3 = tlx.reduce_sum(self.fn(h4) * c1, axis=1)
        sc4 = tlx.reduce_sum(self.fn(h3) * c2, axis=1)
        logits = tlx.concat((sc1, sc2, sc3, sc4), axis=0)

        return logits


class MVGRL(tlx.nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MVGRL, self).__init__()
        self.act = tlx.nn.PRelu()
        self.encoder1 = GCNConv(in_dim, out_dim, add_bias=True)
        self.encoder2 = GCNConv_Dense(in_dim, out_dim, add_bias=True)
        # self.pooling = AvgPooling()
        # self.average = Average()  avgPooling 实现一样的就是dgi的readout
        self.disc = Discriminator(out_dim)
        self.act_fn = tlx.Sigmoid()
        self.loss = tlx.losses.sigmoid_cross_entropy

    def get_embedding(self, edge_index, diff_adj, feat):
        # 输入应该是graph的edge_index, diff_graph的edge_index、feat是graph与diff共用、edge_weight是diff的
        edge_index, norm_weight = calc(edge_index, feat.shape[0])
        h1 = self.encoder1(feat, edge_index, norm_weight, feat.shape[0])
        h2 = self.encoder2(diff_adj, feat)
        h1 = self.act(h1)
        h2 = self.act(h2)

        return h1 + h2
    def forward(self, edge_index, diff_adj, feat, shuf_feat):
        # 输入应该是graph的edge_index, diff_graph的edge_index、feat是graph与diff共用、edge_weight是diff的
        # shuf与dgi一样，都是打乱feat对应的idx即可
        edge_index, norm_weight = calc(edge_index, feat.shape[0])
        h1 = self.act(self.encoder1(feat, edge_index, norm_weight, feat.shape[0]))
        h2 = self.act(self.encoder2(diff_adj, feat))

        h3 = self.act(self.encoder1(shuf_feat, edge_index, norm_weight, feat.shape[0]))
        h4 = self.act(self.encoder2(diff_adj, shuf_feat))

        # summary = tlx.sigmoid(tlx.reduce_mean(pos, axis=0))我理解的dgl的AvgPooling()
        c1 = self.act_fn(tlx.reduce_mean(h1, axis=0))
        c2 = self.act_fn(tlx.reduce_mean(h2, axis=0))
        out = self.disc(h1, h2, h3, h4, c1, c2)
        lbl1 = tlx.ones((feat.shape[0] * 2,))
        lbl2 = tlx.zeros_like(lbl1)
        lbl = tlx.concat((lbl1, lbl2), axis=0)
        return self.loss(out, lbl)






import numpy as np
import scipy.sparse as sp
def calc(edge, num_node):
    weight = np.ones(edge.shape[1])
    sparse_adj = sp.coo_matrix((weight, (edge[0], edge[1])), shape=(num_node, num_node))
    A = (sparse_adj + sp.eye(num_node)).tocoo()
    col, row, weight = A.col, A.row, A.data
    deg = np.array(A.sum(1))
    deg_inv_sqrt = np.power(deg, -0.5).flatten()
    return tlx.convert_to_tensor([col, row], dtype=tlx.int64), tlx.convert_to_tensor(
        np.array(weight * deg_inv_sqrt[col], dtype=np.float32))
