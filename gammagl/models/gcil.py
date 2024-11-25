import torch.nn as nn
import torch.nn.functional as F
from gammagl.models.gcn import GCNModel as GCN

     
class GCILModel(nn.Module):
    r"""GCIL Model proposed in '"Graph Contrastive Invariant Learning from the Causal Perspective"
        <https://arxiv.org/pdf/2401.12564v2.pdf>'_ paper.
            
        Parameters
        ----------
        in_dim: int
            Input feature dimension.
        hid_dim: int
            Hidden layer dimension.
        out_dim: int
            Output feature dimension, usually the number of classes or the desired embedding size.
        n_layers: int
            Number of layers for the GCN model (if GCN is used).
        use_mlp: bool, optional
            If True, the model will use an MLP backbone instead of GCN. Default is False.

    """
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, use_mlp = False):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
        else:
            self.backbone = MLP(in_dim, hid_dim, out_dim)

    def get_embedding(self, graph, feat, attr):
        out = self.backbone(graph, attr, feat)
        return out.detach()

    def forward(self, graph1, feat1, attr1, graph2, feat2, attr2):
        h1 = self.backbone(graph1, attr1, feat1)
        h2 = self.backbone(graph2, attr2, feat2)

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        return z1, z2,h1,h2

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)

        x = self.act_fn(x)
        x = self.layer2(x)

        return x
