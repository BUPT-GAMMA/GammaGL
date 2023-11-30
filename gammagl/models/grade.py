import tensorlayerx as tlx
from gammagl.layers.conv import GCNConv


# Multi-layer Graph Convolutional Networks
class GCN(tlx.nn.Module):
    def __init__(self, in_dim, out_dim, act_fn, num_layers = 2):
        super(GCN, self).__init__()
        assert num_layers>=2
        self.num_layers=num_layers
        self.convs=tlx.nn.ModuleList()

        self.convs.append(GCNConv(in_dim, out_dim*2))
        for _ in range(self.num_layers-2):
            self.convs.append(GCNConv(out_dim*2, out_dim*2))

        self.convs.append(GCNConv(out_dim*2, out_dim))
        self.act_fn=act_fn


    def forward(self, feat, edge_index):
        for i in range(self.num_layers):
            feat=self.act_fn(self.convs[i](feat, edge_index))
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

class GRADE(tlx.nn.Module):
    r"""The  GRAph contrastive learning for DEgree bias (GRADE) model from the
        `"Uncovering the Structural Fairness in Graph Contrastive Learning"
        <https://arxiv.org/abs/2210.03011>`_ paper.

        Parameters
        -----------
        in_dim: int
            Input feature size.
        hid_dim: int
            Hidden feature size.
        out_dim: int
            Output feature size.
        num_layers: int
            Number of the GNN encoder layers.
        act_fn: nn.Module
            Activation function.
        temp: float
            Temperature constant.

     """
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, act_fn, temp):
        super(GRADE, self).__init__()
        self.encoder = GCN(in_dim, hid_dim, act_fn, num_layers)
        self.temp = temp
        self.proj = MLP(hid_dim, out_dim)

    def get_loss(self, z1, z2):

        # calculate SimCLR loss
        f = lambda x: tlx.exp(x / self.temp)
        refl_sim = f(self.get_sim(z1, z1))  # intra-view pairs
        between_sim = f(self.get_sim(z1, z2))  # inter-view pairs

        # between_sim.diag(): positive pairs
        x1 = tlx.reduce_sum(refl_sim, axis=1) + tlx.reduce_sum(between_sim, axis=1) - tlx.diag(refl_sim)
        loss = -tlx.log(tlx.diag(between_sim) / x1)

        return loss

    def get_sim(self, z1, z2):
        '''
            Compute the similarity matix
            '''
        z1 = tlx.ops.l2_normalize(z1, axis=1)
        z2 = tlx.ops.l2_normalize(z2, axis=1)
        return tlx.ops.matmul(z1, tlx.ops.transpose(z2))

    def get_embedding(self, feat, edge):
        # get embeddings from the model for evaluation
        h = self.encoder(feat, edge)
        # h.detach()
        h = tlx.convert_to_tensor(tlx.convert_to_numpy(h), dtype=tlx.float32)
        return h

    def forward(self, feat1, edge1, feat2, edge2):
        # encoding
        h1 = self.encoder(feat1, edge1)
        h2 = self.encoder(feat2, edge2)
        # projection
        z1 = self.proj(h1)
        z2 = self.proj(h2)
        # get loss
        l1 = self.get_loss(z1, z2)
        l2 = self.get_loss(z2, z1)
        ret = (l1 + l2) * 0.5
        return tlx.reduce_mean(ret)


