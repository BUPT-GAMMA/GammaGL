import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.layers.conv import GCNConv
class Encoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_channels, momentum = 0.01)
        self.prelu1 = nn.PRelu()
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels, momentum = 0.01)
        self.prelu2 = nn.PRelu()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.prelu1(self.bn1(x))
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.prelu2(self.bn2(x))
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)
        x = self.layer2(x)
        return x



class EigenMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, period):
        super(EigenMLP, self).__init__()
        self.k = input_dim
        self.period = period
        self.phi = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16))
        self.psi = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
        self.mlp1 = nn.Linear(in_features=2*period, out_features=hidden_dim)
        self.mlp2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, e, u):

        e = e * 100
        period_term = tlx.arange(0, self.period)
        # period_term = period_term.to(u.device)
        period_e = e.unsqueeze(1) * tlx.pow(2, period_term)
        # period_e = period_e.to(u.device)
        fourier_e = tlx.concat([tlx.sin(period_e), tlx.cos(period_e)], axis=-1)
        h = u @ fourier_e
        h = self.mlp1(h)
        h = nn.ReLU()(h)
        h = self.mlp2(h)
        return h


class SpaSpeNode(nn.Module):
    def __init__(self, spa_encoder, spe_encoder, hidden_dim, t=1.):
        super(SpaSpeNode, self).__init__()
        self.t = t
        self.hidden_dim = hidden_dim
        self.spa_encoder = spa_encoder
        self.spe_encoder = spe_encoder
        self.proj = nn.Sequential(nn.Linear(in_features=hidden_dim, out_features=hidden_dim, W_init='xavier_uniform'),
                              nn.PRelu(),
                              nn.Linear(in_features=hidden_dim, out_features=hidden_dim, W_init='xavier_uniform'))

    def forward(self, x, edge_index, e, u, size=-1):
        x_node_spa = self.spa_encoder(x, edge_index)
        x_node_spe = self.spe_encoder(e, u)
        if size > 0:
            x_node_spa = x_node_spa[:size, :]
        h_node_spa = self.proj(x_node_spa)
        h_node_spe = self.proj(x_node_spe)
        return h_node_spa, h_node_spe
