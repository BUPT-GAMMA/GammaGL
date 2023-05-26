import tensorlayerx as tlx
from gammagl.layers.conv import GCNConv


class LogReg(tlx.nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = tlx.nn.Linear(in_features=hid_dim,out_features=out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class MLP(tlx.nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = tlx.nn.Linear(nfeat, nhid)
        self.layer2 = tlx.nn.Linear(nhid, nclass)
        self.bn = tlx.nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = tlx.layers.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act_fn(x)
        x = self.layer2(x)
        return x


class GCN(tlx.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = tlx.nn.ModuleList()

        self.convs.append(GCNConv(in_channels=in_dim, out_channels=hid_dim))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GCNConv(in_channels=hid_dim, out_channels=out_dim))
            self.convs.append(GCNConv(in_channels=hid_dim, out_channels=out_dim))
        self.relu = tlx.layers.ReLU()

    def forward(self, graph, attr, x):
        for i in range(self.n_layers - 1):
            x = self.relu(self.convs[i](x, graph, edge_weight=attr))
        x = self.convs[-1](x, graph, edge_weight=attr)
        return x


class CCA_SSG(tlx.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, use_mlp=False):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
        else:
            self.backbone = MLP(in_dim, hid_dim, out_dim)

    def get_embedding(self, graph, attr, feat):
        out = self.backbone(graph, attr, feat)
        return out.detach()

    def forward(self, graph1, feat1, attr1, graph2, feat2, attr2):
        h1 = self.backbone(graph1, attr1, feat1)
        h2 = self.backbone(graph2, attr2, feat2)

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        return z1, z2
