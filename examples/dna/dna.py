import tensorlayerx as tlx
from dna_conv import DNAConv
import math


class DNAModel(tlx.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads=1, groups=1, name = None):
        super().__init__(name=name)
        self.hidden_channels = hidden_channels
        self.lin1 = tlx.nn.Linear(in_features=in_channels, out_features=hidden_channels, W_init=tlx.nn.he_uniform(a=math.sqrt(5)))
        self.convs = tlx.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                DNAConv(hidden_channels, heads, groups, dropout=0.7))
        self.lin2 = tlx.nn.Linear(in_features=hidden_channels, out_features=out_channels)
        self.relu = tlx.nn.ReLU()
        self.dropout = tlx.nn.Dropout(p=0.7)

    def forward(self, x, edge_index):
        x = self.relu(self.lin1(x))
        x = self.dropout(x)
        x_all = tlx.reshape(x, (-1, 1, self.hidden_channels))
        for conv in self.convs:
            x = self.relu(conv(x_all, edge_index))
            x = tlx.reshape(x, (-1, 1, self.hidden_channels))
            x_all = tlx.concat([x_all, x], axis=1)
        x = x_all[:, -1]
        x = self.dropout(x)
        x = self.lin2(x)

        return tlx.logsoftmax(x, dim=1)
