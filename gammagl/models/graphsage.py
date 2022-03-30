import tensorlayerx as tlx

from gammagl.layers.conv import SAGEConv


class GraphSAGE_Full_Model(tlx.nn.Module):
    def __init__(self,in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE_Full_Model, self).__init__()
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
    
class GraphSAGE_Sample_Model(tlx.nn.Module):
    r""" 
    The GraphSAGE operator from the `"Inductive Representation Learning on
       Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
       Parameters:
        cfg: configuration of GraphSAGE
    """

    def __init__(self, in_feat, hid_feat, out_feat, keep_rate, num_layers, name=None):
        super(GraphSAGE_Sample_Model, self).__init__()
        self.relu = tlx.ReLU()
        self.dropout = tlx.nn.Dropout(keep_rate)
        conv1 = SAGEConv(in_feat, hid_feat)
        conv2 = SAGEConv(hid_feat, out_feat)
        self.num_layer = num_layers
        self.num_class = out_feat
        self.hid_feat = hid_feat
        self.conv = tlx.nn.layers.ModelLayer()
        if num_layers == 1:
            self.conv.append(conv2)
        else:
            self.conv.append(conv1)
            for _ in range(num_layers - 2):
                self.conv.append(SAGEConv(hid_feat, hid_feat))
            self.conv.append(conv2)

    def forward(self, feat, subg):
        h = feat
        for l, (layer, sg) in enumerate(zip(self.conv, subg)):
            target_feat = tf.gather(h, tf.range(0, sg.size[1])) # Target nodes are always placed first.
            h = layer((h, target_feat), sg.edge)
            if l != len(self.conv) - 1:
                h = self.relu(h)
                h = self.dropout(h)
        return h
    def inference(self, feat, dataloader):
        for l, layer in enumerate(self.conv):
            y = tf.zeros((feat.shape[0], self.num_class if l == len(self.conv) - 1 else self.hid_feat))
            for dst_node, all_node, k_hop_ans in dataloader:
                sg = k_hop_ans[0]
                h = tf.gather(feat, all_node)
                target_feat = tf.gather(h, tf.range(0, sg.size[1]))
                h = layer((h, target_feat), sg.edge)
                if l != len(self.conv) - 1:
                    h = self.relu(h)
                    h = self.dropout(h)
                y = tf.tensor_scatter_nd_update(y, dst_node.reshape(-1, 1), h)
            feat = y
        return feat