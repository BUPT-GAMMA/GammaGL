import tensorlayerx as tlx
from gammagl.models import GCNModel as GCN
from gammagl.layers.attention.bga import BGA


class CoBFormer(tlx.nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 gcn_layers: int, layers: int, n_head: int, dropout1=0.5, dropout2=0.1,
                 alpha=0.8, tau=0.5, use_patch_attn=True):
        super(CoBFormer, self).__init__()
        self.alpha = alpha
        self.tau = tau
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.activation = tlx.ReLU()
        self.gcn = GCN(
            feature_dim=in_channels,
            hidden_dim=hidden_channels,
            num_class=out_channels,
            drop_rate=dropout1,
            num_layers=gcn_layers
        )
        self.bga = BGA(num_nodes, in_channels, hidden_channels, out_channels, layers, n_head,
                       use_patch_attn, dropout1, dropout2)
        self.attn = None

    def forward(self, x, patch, edge_index, edge_weight=None, num_nodes=None, need_attn=False):
        z1 = self.gcn(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
        z2 = self.bga(x, patch, need_attn)
        
        if need_attn:
            self.attn = self.bga.attn

        return z1, z2
