import tensorlayerx as tlx
from gammagl.layers.conv import GANConv

class GANModel(tlx.nn.Module):
    def __init__(self,
                 feature_dim,
                 hidden_dim,
                 num_class,
                 drop_rate,
                 agg='sum',
                 num_layers=2,
                 name=None):
        super().__init__(name=name)
        self.node_encoder = tlx.layers.Linear(out_features=hidden_dim, in_features=feature_dim)
        self.layers = tlx.nn.ModuleList()
        self.norm = tlx.nn.BatchNorm1d()
        self.act = tlx.nn.ReLU()
        self.drop = tlx.nn.Dropout(drop_rate)
        self.num_layers = num_layers
        for i in range(num_layers):
            self.layers.append(GANConv(in_channels=hidden_dim, out_channels=hidden_dim, agg=agg))

        self.lin = tlx.layers.Linear(out_features=num_class, in_features=hidden_dim)


    def forward(self,x ,edge_index, edge_weight, num_nodes):

        x = self.node_encoder(x)
        for i in range(self.num_layers):
            x = self.norm(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.layers[i](x, edge_index, edge_weight, num_nodes)
        return self.lin(x)


