import tensorlayerx as tlx
from gammagl.layers.conv import SimpleHGNConv


class SimpleHGNModel(tlx.nn.Module):
    r'''This is a model SimpleHGN from `Are we really making much progress? Revisiting, benchmarking, and refining heterogeneous graph neural networks
    <https://dl.acm.org/doi/pdf/10.1145/3447548.3467350>`__

    Parameters
    ----------
    feature_dims: list
        Dimension list of feature vectors in original input.
    hidden_dim: int
        Dimension of feature vector in AGNN.
    edge_dim:
        The edge dimension
    heads_list: list
        The list of the number of heads in each layer
    num_etypes: int
        The number of the edge type
    num_classes: int
        The number of the output classes
    num_layers: int
        The number of layers we used
    activation:
        Activation function we used
    feat_drop: float
        The feature drop rate
    attn_drop: float
        The attention score drop rate
    negative_slope: float
        The negative slope used in the LeakyReLU
    residual: boolean
        Whether we need the residual operation
    beta: float
        The hyperparameter used in edge residual
    
    '''
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

        self.fc_list = tlx.nn.ModuleList([tlx.nn.Linear(hidden_dim, in_features=feature_dim, W_init=tlx.initializers.XavierNormal(gain=1.414)) for feature_dim in feature_dims ])

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
        x = [fc(feature) for fc, feature in zip(self.fc_list, x)]
        x = tlx.ops.concat(x, axis=0)

        alpha = None
        for l in range(self.num_layers):
            x, alpha = self.hgn_layers[l](x, edge_index, e_feat, alpha)
            x = tlx.reshape(x, (-1, self.heads_list[l] * self.hidden_dim))

        x, _ = self.hgn_layers[-1](x, edge_index, e_feat, alpha)
        x = tlx.ops.reduce_mean(x, axis=1)
        x = tlx.ops.l2_normalize(x, axis=-1)
        return x
