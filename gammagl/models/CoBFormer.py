import torch

from gammagl.models.ffn import *
from gammagl.models.gcn import *
from gammagl.layers.attention.BGA import BGA


class CoBFormer(torch.nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 activation, gcn_layers: int, gcn_type: int, layers: int, n_head: int, dropout1=0.5, dropout2=0.1,
                 alpha=0.8, tau=0.5, gcn_use_bn=False, use_patch_attn=True):
        super(CoBFormer, self).__init__()
        self.alpha = alpha
        self.tau = tau
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.activation = activation
        self.dropout = nn.Dropout(dropout1)
        if gcn_type == 1:
            self.gcn = GCN(in_channels, hidden_channels, out_channels, activation, k=gcn_layers, use_bn=gcn_use_bn)
        else:
            self.gcn = GraphConv(in_channels, hidden_channels, out_channels, num_layers=gcn_layers, use_bn=gcn_use_bn)
        # self.gat = GAT(in_channels, hidden_channels, out_channels, activation, k=gcn_layers, use_bn=gcn_use_bn)
        self.bga = BGA(num_nodes, in_channels, hidden_channels, out_channels, layers, n_head,
                                         use_patch_attn, dropout1, dropout2)
        self.attn = None

    def forward(self, x: torch.Tensor, patch: torch.Tensor, edge_index: torch.Tensor, need_attn=False):
        z1 = self.gcn(x, edge_index)
        z2 = self.bga(x, patch, need_attn)
        if need_attn:
            self.attn = self.beyondformer.attn

        return z1, z2

    def loss(self, pred1, pred2, label, mask):
        l1 = F.cross_entropy(pred1[mask], label[mask])
        l2 = F.cross_entropy(pred2[mask], label[mask])
        pred1 *= self.tau
        pred2 *= self.tau
        l3 = F.cross_entropy(pred1[~mask], F.softmax(pred2, dim=1)[~mask])
        l4 = F.cross_entropy(pred2[~mask], F.softmax(pred1, dim=1)[~mask])
        loss = self.alpha * (l1 + l2) + (1 - self.alpha) * (l3 + l4)
        return loss