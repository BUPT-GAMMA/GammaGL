import tensorlayerx as tlx
import tensorlayerx.nn as nn
from tensorlayerx import relu
from gammagl.layers.conv.pna_conv import PNAConv
from gammagl.layers.pool.glob import global_sum_pool


class PNAModel(nn.Module):
    def __init__(self, in_channels, out_channels, aggregators, scalers, deg, edge_dim, towers, pre_layers, post_layers, divide_input):
        super().__init__()

        self.node_emb = nn.Embedding(21, 75)
        self.edge_emb = nn.Embedding(4, 50)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        aggregators = aggregators.split()
        scalers = scalers.split()

        for _ in range(4):
            conv = PNAConv(in_channels=in_channels, out_channels=out_channels,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=edge_dim, towers=towers, pre_layers=pre_layers, post_layers=post_layers,
                           divide_input=divide_input)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(num_features=out_channels))

        self.mlp = nn.Sequential(nn.Linear(in_features=out_channels, out_features=50, act=tlx.ReLU),
                                 nn.Linear(in_features=50, out_features=25, act=tlx.ReLU),
                                 nn.Linear(in_features=25, out_features=1))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(tlx.squeeze(x=x, axis=1))
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_sum_pool(x, batch)
        return self.mlp(x)
