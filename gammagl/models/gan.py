import tensorlayerx as tlx
from gammagl.layers.conv import GANConv
from .deepgcn import DeepGCNLayer

class GANModel(tlx.nn.Module):
    def __init__(self,
                 feature_dim,
                 hidden_dim,
                 edge_dim,
                 num_class,
                 drop_rate,
                 agg='sum',
                 num_layers=2,
                 name=None):
        super().__init__(name=name)
        self.drop_rate = drop_rate
        self.num_layers = num_layers
        self.node_encoder = tlx.layers.Linear(out_features=hidden_dim, in_features=feature_dim,
                                              W_init='xavier_uniform')
        self.edge_encoder = tlx.layers.Linear(out_features=hidden_dim, in_features=edge_dim,
                                              W_init='xavier_uniform')

        self.layers = tlx.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GANConv(in_channels=hidden_dim, out_channels=hidden_dim, agg=agg)
            norm = tlx.nn.LayerNorm(hidden_dim)
            act = tlx.nn.ReLU()
            layer = DeepGCNLayer(conv, norm, act, dropout=drop_rate)
            self.layers.append(layer)
        self.drop = tlx.nn.Dropout(drop_rate)
        self.lin = tlx.layers.Linear(out_features=num_class, in_features=hidden_dim,
                                     W_init='xavier_uniform')

    def forward(self, x, edge_index, edge_weight, num_nodes):
        x = self.node_encoder(x)
        edge_weight = self.edge_encoder(edge_weight)

        x = self.layers[0].conv(x, edge_index, edge_weight, num_nodes)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_weight, num_nodes)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = self.drop(x)
        return self.lin(x)



