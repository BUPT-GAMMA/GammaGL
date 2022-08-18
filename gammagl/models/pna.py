import tensorlayerx as tlx
import tensorlayerx.nn as nn
from tensorlayerx import relu
from gammagl.layers.conv.pna_conv import PNAConv
from gammagl.layers.pool.glob import global_sum_pool


class PNAModel(nn.Module):
    def __init__(self, deg):
        super().__init__()

        self.node_emb = nn.Embedding(21, 75)
        self.edge_emb = nn.Embedding(4, 50)
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(4):
            conv = PNAConv(in_channels=75, out_channels=75,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=50, towers=5, pre_layers=1, post_layers=1,
                             divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(num_features=75))

        self.mlp = nn.Sequential(nn.Linear(in_features=75, out_features=50, act=tlx.ReLU),
                                 nn.Linear(in_features=50, out_features=25, act=tlx.ReLU),
                                 nn.Linear(in_features=25, out_features=1))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(tlx.squeeze(x=x, axis=1))
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_sum_pool(x, batch)
        return self.mlp(x)