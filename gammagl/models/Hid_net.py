import tensorlayerx as tlx

from gammagl.layers.conv import Hid_conv
class hid_net(tlx.nn.Module):
    def __init__(self, in_feats,
                 n_hidden,
                 n_classes,
                 k,
                 alpha,
                 beta,
                 gamma,
                 bias,
                 normalize,
                 add_self_loops,
                 drop,
                 dropout,
                 sigma1,
                 sigma2):
        super(hid_net, self).__init__()
        self.convs = tlx.nn.ModuleList()
        self.convs.append(Hid_conv(in_feats,
                 n_hidden,
                 n_classes,
                 k,
                 alpha,
                 beta,
                 gamma,
                 bias,
                 normalize,
                 add_self_loops,
                 drop,
                 dropout,
                 sigma1,
                 sigma2))
        
    def forward(self, feat, edge, edge_weight=None,num_nodes=0):
        h = feat
        for l, layer in enumerate(self.convs):
            h = layer(h, edge,edge_weight,num_nodes)
        return h