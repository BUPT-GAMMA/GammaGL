from gammagl.models.ffn import *
from gammagl.models.gcn import *
from gammagl.layers.attention.BGA_layer import BGALayer
from torch_geometric.nn import GCNConv
import torch.nn as nn


class BGA(torch.nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 layers: int, n_head: int, use_patch_attn=True, dropout1=0.5, dropout2=0.1, need_attn=False):
        super(BGA, self).__init__()
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout1)
        self.attribute_encoder = FFN(in_channels, hidden_channels)
        self.BGALayers = nn.ModuleList()
        for _ in range(0, layers):
            self.BGALayers.append(
                BGALayer(n_head, hidden_channels, use_patch_attn, dropout=dropout2))
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.attn=[]

    def forward(self, x: torch.Tensor, patch: torch.Tensor, need_attn=False):
        patch_mask = (patch != self.num_nodes - 1).float().unsqueeze(-1)
        attn_mask = torch.matmul(patch_mask, patch_mask.transpose(1, 2)).int()

        x = self.attribute_encoder(x)
        for i in range(0, self.layers):
            x = self.BGALayers[i](x, patch, attn_mask, need_attn)
            if need_attn:
                self.attn.append(self.BGALayers[i].attn)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
