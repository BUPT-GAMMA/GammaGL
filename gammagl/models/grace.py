import tensorlayerx as tlx
import numpy as np
import scipy.sparse as sp
from gammagl.utils.loop import add_self_loops

from gammagl.layers.conv import GCNConv


def calc(edge, num_node):
    edge, _ = add_self_loops(edge, 1)
    edge_weight = tlx.ones((edge.shape[1],))
    src, dst = edge[0], edge[1]
    deg = tlx.reshape(tlx.ops.unsorted_segment_sum(tlx.reshape(edge_weight, (-1, 1)), src, num_segments=num_node),
                      (-1,))  # tlx更新后可以去掉 reshape
    deg_inv_sqrt = tlx.pow(deg, -0.5)
    # deg_inv_sqrt[tlx.is_inf(deg_inv_sqrt)] = 0 # may exist solo node
    weights = tlx.ops.gather(deg_inv_sqrt, src) * edge_weight * tlx.ops.gather(deg_inv_sqrt, dst)
    return edge, weights
    # weight = np.ones(edge.shape[1])
    # sparse_adj = sp.coo_matrix((weight, (edge[0], edge[1])), shape=(num_node, num_node))
    # A = (sparse_adj + sp.eye(num_node)).tocoo()
    # col, row, weight = A.col, A.row, A.data
    # deg = np.array(A.sum(1))
    # deg_inv_sqrt = np.power(deg, -0.5).flatten()
    # return col, row, np.array(deg_inv_sqrt[row] * weight * deg_inv_sqrt[col], dtype=np.float32)



class GCN(tlx.nn.Module):
    def __init__(self, in_feat, hid_feat, num_layers, activation):
        super(GCN, self).__init__()
        assert num_layers >= 2
        self.num_layers = num_layers
        # self.convs = tlx.nn.Sequential()
        cur = []

        cur.append(GCNConv(in_channels=in_feat, out_channels=hid_feat))
        for _ in range(num_layers - 1):
            cur.append(GCNConv(in_channels=hid_feat, out_channels=hid_feat))
        self.convs = tlx.nn.Sequential(cur)
        self.act = activation

    def forward(self, feat, edge_index, edge_weight, num_nodes):
        for i in range(self.num_layers):
            feat = self.act(self.convs[i](feat, edge_index, edge_weight, num_nodes))
        return feat


# Multi-layer(2-layer) Perceptron
class MLP(tlx.nn.Module):
    def __init__(self, in_feat, out_feat):
        super(MLP, self).__init__()

        self.fc1 = tlx.nn.Linear(in_features=in_feat, out_features=out_feat)
        self.fc2 = tlx.nn.Linear(in_features=out_feat, out_features=in_feat)

    def forward(self, x):
        x = tlx.elu(self.fc1(x))
        return self.fc2(x)


class grace(tlx.nn.Module):
    def __init__(self, in_feat, hid_feat, out_feat, num_layers, activation, temp):
        super(grace, self).__init__()
        self.encoder = GCN(in_feat, hid_feat, num_layers, activation)
        self.temp = temp
        self.proj = MLP(hid_feat, out_feat)

    def get_loss(self, z1, z2):
        # calculate SimCLR loss
        f = lambda x: tlx.exp(x / self.temp)
        refl_sim = f(self.sim(z1, z1))  # intra-view pairs
        between_sim = f(self.sim(z1, z2))  # inter-view pairs

        # between_sim.diag(): positive pairs
        x1 = tlx.reduce_sum(refl_sim, axis=1) + tlx.reduce_sum(between_sim, axis=1) - tlx.diag(refl_sim, 0)
        loss = -tlx.log(tlx.diag(between_sim, 0) / x1)

        return loss

    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = tlx.ops.l2_normalize(z1, axis=1)
        z2 = tlx.ops.l2_normalize(z2, axis=1)
        return tlx.matmul(z1, tlx.transpose(z2))

    def get_embeding(self, feat, edge, weight, num_nodes):
        h = self.encoder(feat, edge, weight, num_nodes)
        return h

    def forward(self, graph1, graph2):
        # encoding
        h1 = self.encoder(graph1.x, graph1.edge_index, graph1.edge_weight, graph1.num_nodes)
        h2 = self.encoder(graph2.x, graph2.edge_index, graph2.edge_weight, graph2.num_nodes)
        # projection
        z1 = self.proj(h1)
        z2 = self.proj(h2)
        # get loss
        l1 = self.get_loss(z1, z2)
        l2 = self.get_loss(z2, z1)
        ret = (l1 + l2) / 2
        return tlx.reduce_mean(ret)