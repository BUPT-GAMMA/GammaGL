import tensorlayerx as tlx

from gammagl.layers.conv import SAGEConv


class GraphSAGEModel(tlx.nn.Module):
    def __init__(self,in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGEModel, self).__init__()
        self.convs = tlx.nn.SequentialLayer()
        self.dropout = tlx.nn.Dropout(dropout)
        self.activation = activation
        self.n_layers = n_layers
        # input layer
        self.convs.append(SAGEConv(in_feats, n_hidden, activation, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.convs.append(SAGEConv(n_hidden, n_hidden, activation, aggregator_type))
        # output layer
        self.convs.append(SAGEConv(n_hidden, n_classes, None, aggregator_type))  # activation None

    def forward(self, feat, edge):
        h = self.dropout(feat)
        for l, layer in enumerate(self.convs):
            h = layer(h, edge)
            if l < self.n_layers:
                h = self.activation(h)
                h = self.dropout(h)
        return h