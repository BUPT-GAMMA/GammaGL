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
    # def __init__(self, in_feats,n_classes,**kwargs):
        super(hid_net, self).__init__()
        self.convs = tlx.nn.ModuleList()
        # self.dropout = tlx.nn.Dropout(dropout)
        # input layer
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
        # hidden layers
        # for i in range(n_layers - 1):
        #     self.convs.append(Hid_conv(n_hidden, n_hidden, dropout,bias ,cached, add_self_loops, normalize,drop,sigma1,sigma2 ,**kwargs))
        # output layer
        # self.convs.append(Hid_conv(n_hidden, n_hidden,n_classes,dropout,bias ,cached, add_self_loops, normalize,drop,sigma1,sigma2 ,**kwargs))  # activation None

    def forward(self, feat, edge, edge_weight=None,num_nodes=0):
        h = feat
        for l, layer in enumerate(self.convs):
            h = layer(h, edge,edge_weight,num_nodes)
            # if l != len(self.convs) - 1:
            #     h = self.dropout(h)
        return h