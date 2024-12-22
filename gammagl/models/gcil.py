import tensorlayerx.nn as nn
import tensorlayerx as tlx
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
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, use_mlp=False, drop_rate=0.2):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, out_dim, drop_rate=drop_rate, num_layers=n_layers)
        else:
            self.backbone = MLP(in_dim, hid_dim, out_dim)
            

    def get_embedding(self, graph, feat, attr):
        x = graph.x
        num_nodes = graph.num_nodes
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight if hasattr(graph, 'edge_weight') else None
        out = self.backbone(x, edge_index, edge_weight, num_nodes)
        return out.detach()

    def forward(self, graph1, feat1, attr1, graph2, feat2, attr2):
        x1 = graph1.x  
        x2 = graph2.x 
        
        num_nodes1 = graph1.num_nodes
        num_nodes2 = graph2.num_nodes
        
        edge_index1 = graph1.edge_index 
        edge_index2 = graph2.edge_index 
        
        edge_weight1 = graph1.edge_weight if hasattr(graph1, 'edge_weight') else None
        edge_weight2 = graph2.edge_weight if hasattr(graph2, 'edge_weight') else None
        

        h1 = self.backbone(x1, edge_index1, edge_weight1, num_nodes1)
        h2 = self.backbone(x2, edge_index2, edge_weight2, num_nodes2)

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        return z1, z2, h1, h2

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        # self.fc = nn.Linear(hid_dim, out_dim)
        self.fc = nn.Linear(in_features=hid_dim, out_features=out_dim)

    def forward(self, x):
        z = self.fc(x)
        return z


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