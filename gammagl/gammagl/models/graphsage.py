import tensorlayerx as tlx
from typing import Tuple, List

from gammagl.layers.conv import SAGEConv


class GraphSAGE_Full_Model(tlx.nn.Module):
    def __init__(self, in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE_Full_Model, self).__init__()
        self.convs = tlx.nn.ModuleList()
        self.dropout = tlx.nn.Dropout(dropout)
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
            if l != len(self.convs) - 1:
                h = self.dropout(h)
        return h


class GraphSAGE_Sample_Model(tlx.nn.Module):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on Large Graphs"
        <https://arxiv.org/abs/1706.02216>`_ paper

        Parameters
        ----------
        in_feat:
            number of input feature.
        hid_feat:
            number of hidden feature.
        out_feat:
            number of output feature.
        drop_rate:
            dropout rate.
        num_layers:
            number of sage layers.
        name:
            model name.

    """

    def __init__(self, in_feat, hid_feat, out_feat, drop_rate, num_layers, name=None):
        super(GraphSAGE_Sample_Model, self).__init__()
        self.convs = tlx.nn.ModuleList()
        self.dropout = tlx.nn.Dropout(drop_rate)

        self.num_layer = num_layers
        self.num_class = out_feat
        self.hid_feat = hid_feat

        if num_layers == 1:
            conv = SAGEConv(hid_feat, out_feat)
            self.convs.append(conv)
        else:
            conv1 = SAGEConv(in_feat, hid_feat, tlx.ReLU())
            conv2 = SAGEConv(hid_feat, out_feat)
            self.convs.append(conv1)
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hid_feat, hid_feat, tlx.ReLU()))
            self.convs.append(conv2)

    def forward(self, x, edgeIndices):
        for l, (layer, edgeIndex) in enumerate(zip(self.convs, edgeIndices)):
            target_x = tlx.gather(x, tlx.arange(0, edgeIndex.size[1]))  # Target nodes are always placed first.
            x = layer((x, target_x), edgeIndex.edge_index)
            if l != len(self.convs) - 1:
                x = self.dropout(x)
        return x

    def inference(self, feat, dataloader, cur_x):
        for l, layer in enumerate(self.convs):
            y = tlx.zeros((feat.shape[0], self.num_class if l == len(self.convs) - 1 else self.hid_feat))
            for dst_node, n_id, adjs in dataloader:
                if isinstance(adjs, (List, Tuple)):
                    sg = adjs[0]
                else:
                    sg = adjs
                h = tlx.gather(feat, n_id)
                target_feat = tlx.gather(h, tlx.arange(0, sg.size[1]))
                h = layer((h, target_feat), sg.edge_index)
                # if l != len(self.convs) - 1:
                #     h = self.dropout(h)
                # soon will compile
                y = tlx.tensor_scatter_nd_update(y, tlx.reshape(dst_node, shape=(-1, 1)), h)
            feat = y
        return feat
