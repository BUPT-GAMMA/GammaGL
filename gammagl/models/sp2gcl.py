import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.layers.conv import GCNConv


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_channels, momentum=0.01)
        self.prelu1 = nn.PRelu()
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_channels, momentum=0.01)
        self.prelu2 = nn.PRelu()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.prelu1(self.bn1(x))
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.prelu2(self.bn2(x))
        return x


class EigenMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, period):
        super(EigenMLP, self).__init__()

        self.period = period

        self.phi = nn.Sequential(nn.Linear(in_features=1, out_features=hidden_dim), nn.ReLU(),
                                 nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.psi = nn.Sequential(nn.Linear(in_features=hidden_dim, out_features=hidden_dim), nn.ReLU(),
                                 nn.Linear(in_features=hidden_dim, out_features=1))
        self.mlp = nn.Sequential(nn.Linear(in_features=2 * period, out_features=hidden_dim), nn.ReLU(),
                                 nn.Linear(in_features=hidden_dim, out_features=hidden_dim))

        self.relu = nn.ReLU()

    def forward(self, e, u):
        u = tlx.expand_dims(u, axis=-1)
        u_transformed = self.psi(self.phi(u) + self.phi(-u))
        u = tlx.ops.squeeze(u_transformed, axis=-1)

        # e = e * 100
        period_term = tlx.arange(1, self.period + 1)
        e_unsqueeze = tlx.expand_dims(e, axis=1)
        fourier_e = tlx.reshape(tlx.stack((tlx.sin(e_unsqueeze * period_term), tlx.cos(e_unsqueeze * period_term)), axis=1), (-1, self.period * 2))

        h = tlx.matmul(u, fourier_e)
        h = self.mlp(h)
        return h


class SpaSpeNode(nn.Module):
    def __init__(self, input_dim, spe_dim, hidden_dim, output_dim, period, name = None):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.spe_dim = spe_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.period = period

        self.spa_encoder = Encoder(self.input_dim, self.hidden_dim)
        self.spe_encoder = EigenMLP(self.spe_dim, self.hidden_dim, self.period)

        self.spa_projection_head = nn.Sequential(
                                        nn.Linear(in_features=hidden_dim, out_features=output_dim, W_init='xavier_uniform'),
                                        nn.PRelu(),
                                        nn.Linear(in_features=output_dim, out_features=output_dim, W_init='xavier_uniform')
                                    )

        self.spe_projection_head = nn.Sequential(
                                        nn.Linear(in_features=hidden_dim, out_features=output_dim, W_init='xavier_uniform'),
                                        nn.PRelu(),
                                        nn.Linear(in_features=output_dim, out_features=output_dim, W_init='xavier_uniform')
                                    )

    def forward(self, x, edge_index, e, u):
        x_node_spa = self.spa_encoder(x, edge_index)
        x_node_spe = self.spe_encoder(e, u)

        h_node_spa = self.spa_projection_head(x_node_spa)
        h_node_spe = self.spe_projection_head(x_node_spe)

        h1 = tlx.l2_normalize(h_node_spa, axis=-1, eps=1e-12)
        h2 = tlx.l2_normalize(h_node_spe, axis=-1, eps=1e-12)

        return h1, h2
