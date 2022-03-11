import tensorlayerx as tl
import tensorflow as tf
import numpy as np
from gammagl.layers.conv import SAGEConv

class GraphSAGEModel(tl.nn.Module):
    r""" 
    The GraphSAGE operator from the `"Inductive Representation Learning on
       Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
       Parameters:
        cfg: configuration of GraphSAGE
    """

    def __init__(self, cfg, name=None):
        super().__init__(name=name)
        self.relu = tl.ReLU()
        self.dropout = tl.Dropout(cfg.keep_rate)
        conv1 = SAGEConv(cfg.feature_dim, cfg.hidden_dim)
        conv2 = SAGEConv(cfg.hidden_dim, cfg.num_class)
        self.num_layer = cfg.num_layer
        self.num_class = cfg.num_class
        self.hid_feat = cfg.hidden_dim
        self.conv = tl.layers.SequentialLayer()
        self.conv.append(conv1)
        for _ in range(cfg.num_layer - 2):
            self.conv.append(SAGEConv(cfg.hidden_dim, cfg.hidden_dim))
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


