import math
import random
import tensorlayerx as tlx
from gammagl.layers.conv import MAGCLConv


class Encoder(tlx.nn.Module):

    def __init__(self, in_feat, out_feat, iter_K, activation, name=None, ):
        super().__init__(name=name)
        self.activation = activation

        self.conv = tlx.nn.ModuleList()

        self.conv.append(MAGCLConv(in_channels=in_feat, out_channels=out_feat))
        self.conv.append(MAGCLConv(in_channels=out_feat, out_channels=out_feat))

        stdv = math.sqrt(6.0 / (in_feat + out_feat))
        self.weight = tlx.random_uniform((in_feat, out_feat), -stdv, stdv)

    def forward(self, x, edge_index, edge_weight, num_nodes, R=[1, 2]):

        K1 = random.randint(0, R[0])
        K2 = random.randint(0, R[1])

        x = self.activation(self.conv[0](x, edge_index, K1, edge_weight, num_nodes))
        x = self.conv[1](x, edge_index, K2, edge_weight, num_nodes)

        return self.activation(x)


# Multi-layer(2-layer) Perceptron
class MLP(tlx.nn.Module):
    def __init__(self, in_feat, out_feat):
        super(MLP, self).__init__()
        # init = tlx.nn.initializers.HeNormal(a=math.sqrt(5))
        self.fc1 = tlx.nn.Linear(in_features=in_feat, out_features=out_feat)  # , W_init=init)
        self.fc2 = tlx.nn.Linear(in_features=out_feat, out_features=in_feat)  # , W_init=init)

    def forward(self, x):
        x = tlx.elu(self.fc1(x))
        return self.fc2(x)


class NewGrace(tlx.nn.Module):
    def __init__(self, in_feat, hid_feat, out_feat, num_layers, activation, temp=0.5):
        super(NewGrace, self).__init__()
        self.encoder = Encoder(in_feat, out_feat, num_layers, activation)
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

    def forward(self,
                x1, edge_index1, edge_weight1, num_nodes1,
                x2, edge_index2, edge_weight2, num_nodes2,
                R1=[1,2], R2=[1,2]):
        # encoding
        h1 = self.encoder(x1, edge_index1, edge_weight1, num_nodes1, R1)
        h2 = self.encoder(x2, edge_index2, edge_weight2, num_nodes2, R2)
        # projection
        z1 = self.proj(h1)
        z2 = self.proj(h2)
        # get loss
        l1 = self.get_loss(z1, z2)
        l2 = self.get_loss(z2, z1)
        ret = (l1 + l2) * 0.5
        return tlx.reduce_mean(ret)

