import tensorlayerx as tlx
from gammagl.layers.conv import SimpleHGNConv
def TODO():
    return

class SimpleHGN(tlx.nn.Module):

    def __init__(self,
                feature_dims,
                hidden_dim,
                edge_dim,
                heads_list,
                num_etypes,
                num_classes,
                num_layers,
                activation,
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                beta
                ):
        TODO()
        self.num_layers = num_layers
        self.hgn_layers = tlx.nn.ModuleList()
        self.activation = activation
        self.fc_list = tlx.nn.ModuleList([tlx.nn.Linear(hidden_dim, W_init=tlx.initializers.XavierNormal(gain=1.414))] for _ in feature_dims)

        self.hgn_layers.append(SimpleHGNConv(in_feats=hidden_dim, 
                                            out_feats=hidden_dim, 
                                            edge_types=num_etypes, 
                                            edge_feats=edge_dim, 
                                            heads=heads_list[0]))
        for l in range(1,num_layers):
            self.hgn_layers.append(SimpleHGNConv(in_feats=hidden_dim * heads_list[l-1], 
                                                out_feats=hidden_dim, 
                                                edge_types=num_etypes, 
                                                edge_feats=edge_dim, 
                                                heads=heads_list[l]))

        hgn_layers.append(SimpleHGNConv(in_feats=hidden_dim*heads_list[-2], 
                                            out_feats=num_classes, 
                                            edge_types=num_etypes, 
                                            edge_feats=edge_dim, 
                                            heads=heads_list[-1]))

    def forward(self, x, edge_index):
        TODO()
        x = [ fc(feature) for fc, feature in zip(self.fc_list, x)]
        x = tlx.convert_to_tensor(x)

        res_attn = None
        for l in range(self.num_layers):
            x, alpha = self.hgn_layers[l]()

        x, _ = self.hgn_layers[-1]()

        x = tlx.ops.l2_normalize(x, axis=-1)
        return x
