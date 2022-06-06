import tensorlayerx as tlx
from gammagl.layers.conv import SimpleHGNConv
def TODO():
    return

class SimpleHGNModel(tlx.nn.Module):

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
        super().__init__()
        self.num_layers = num_layers
        self.hgn_layers = tlx.nn.ModuleList()
        self.heads_list = heads_list
        self.hidden_dim = hidden_dim
        #self.activation = activation
        self.fc_list = tlx.nn.ModuleList([tlx.nn.Linear(hidden_dim, in_features=feature_dim, W_init=tlx.initializers.XavierNormal(gain=1.414)) for feature_dim in feature_dims ])
        #for _ in feature_dims:
        #    self.fc_list.append()

        self.hgn_layers.append(SimpleHGNConv(in_feats=hidden_dim, 
                                            out_feats=hidden_dim, 
                                            num_etypes=num_etypes, 
                                            edge_feats=edge_dim, 
                                            heads=heads_list[0],
                                            feat_drop=feat_drop,
                                            attn_drop=attn_drop,
                                            negative_slope=negative_slope,
                                            activation=activation,
                                            residual=False,
                                            beta=beta))
        for l in range(1,num_layers):
            self.hgn_layers.append(SimpleHGNConv(in_feats=hidden_dim * heads_list[l-1], 
                                                out_feats=hidden_dim, 
                                                num_etypes=num_etypes, 
                                                edge_feats=edge_dim, 
                                                heads=heads_list[l],
                                                feat_drop=feat_drop,
                                                attn_drop=attn_drop,
                                                negative_slope=negative_slope,
                                                activation=activation,
                                                residual=residual,
                                                beta=beta))

        self.hgn_layers.append(SimpleHGNConv(in_feats=hidden_dim*heads_list[-2], 
                                        out_feats=num_classes, 
                                        num_etypes=num_etypes, 
                                        edge_feats=edge_dim, 
                                        heads=heads_list[-1],
                                        feat_drop=feat_drop,
                                        attn_drop=attn_drop,
                                        negative_slope=negative_slope,
                                        residual=residual,
                                        beta=beta))

    def forward(self, x, edge_index, e_feat):
        #将不同节点的维度统一
        x = [ fc(feature) for fc, feature in zip(self.fc_list, x)]
        x = tlx.ops.concat(x, axis=0)

        alpha = None
        for l in range(self.num_layers):
            x, alpha = self.hgn_layers[l](x, edge_index, e_feat)
            x = tlx.reshape(x, (-1, self.heads_list[l] * self.hidden_dim))

        x, _ = self.hgn_layers[-1](x, edge_index, e_feat)
        x = tlx.ops.reduce_mean(x, axis=1)
        x = tlx.ops.l2_normalize(x, axis=-1)
        return x
