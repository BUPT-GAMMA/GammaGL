import math
import tensorlayerx as tlx
from gammagl.layers.conv import GCNConv

class GCN(tlx.nn.Module):
    def __init__(self, in_feat, hid_feat, num_layers, activation):
        super(GCN, self).__init__()
        assert num_layers >= 2
        self.num_layers = num_layers
        self.convs = tlx.nn.ModuleList()


        self.convs.append(GCNConv(in_channels=in_feat, out_channels=hid_feat))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(in_channels=hid_feat, out_channels=hid_feat))

        self.act = activation

    def forward(self, feat, edge_index, edge_weight, num_nodes):
        for i in range(self.num_layers):
            feat = self.act(self.convs[i](feat, edge_index, edge_weight, num_nodes))
        return feat


# Multi-layer(2-layer) Perceptron
class MLP(tlx.nn.Module):
    def __init__(self, in_feat, out_feat):
        super(MLP, self).__init__()
        # init = tlx.nn.initializers.HeNormal(a=math.sqrt(5))
        self.fc1 = tlx.nn.Linear(in_features=in_feat, out_features=out_feat) #, W_init=init)
        self.fc2 = tlx.nn.Linear(in_features=out_feat, out_features=in_feat) # , W_init=init)

    def forward(self, x):
        x = tlx.elu(self.fc1(x))
        return self.fc2(x)


class GraceModel(tlx.nn.Module):
    def __init__(self, in_feat, hid_feat, out_feat, num_layers, activation, temp):
        super(GraceModel, self).__init__()
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

        z1 = tlx.l2_normalize(z1, axis=1)
        z2 = tlx.l2_normalize(z2, axis=1)

        return tlx.matmul(z1, tlx.transpose(z2))

    def get_embeding(self, feat, edge, weight, num_nodes):
        h = self.encoder(feat, edge, weight, num_nodes)
        return h

    # def forward(self, graph1, graph2):
    def forward(self, feat1, edge1, weight1, num_node1, feat2, edge2, weight2, num_node2):
        # encoding
        h1 = self.encoder(feat1, edge1, weight1, num_node1)
        h2 = self.encoder(feat2, edge2, weight2, num_node2)
        # projection
        z1 = self.proj(h1)
        z2 = self.proj(h2)
        # get loss
        l1 = self.get_loss(z1, z2)
        l2 = self.get_loss(z2, z1)
        ret = (l1 + l2) * 0.5
        return tlx.reduce_mean(ret)
