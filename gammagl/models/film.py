import tensorlayerx as tlx
from gammagl.layers.conv.film_conv import FILMConv


class FILMModel(tlx.nn.Module):
    def __init__(self, in_channels,
                 hidden_dim,
                 out_channels,
                 drop_rate,
                 name=None):
        super(FILMModel, self).__init__(name=name)

        self.convs = tlx.nn.ModuleList()
        self.convs.append(FILMConv(in_channels=in_channels, out_channels=hidden_dim))
        for _ in range(2):
            self.convs.append(FILMConv(in_channels=hidden_dim, out_channels=hidden_dim))
        self.convs.append(FILMConv(in_channels=hidden_dim, out_channels=out_channels, act=None))

        self.norms = tlx.nn.ModuleList()
        for _ in range(3):
            self.norms.append(tlx.nn.BatchNorm1d(momentum=0.1))

        self.dropout = tlx.nn.Dropout(drop_rate)

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return x
