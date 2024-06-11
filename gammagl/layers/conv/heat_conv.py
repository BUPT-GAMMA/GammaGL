import numpy as np
import tensorlayerx as tlx
from .message_passing import MessagePassing
from tensorlayerx.nn import Linear, LeakyReLU, Dropout


class HEATlayer(MessagePassing):
    r"""
       The heterogeneous edge-enhanced graph attention operator from the`"Heterogeneous Edge-Enhanced Graph Attention Network For Multi-Agent Trajectory Prediction"
       <https://arxiv.org/abs/2106.07161>`_ paper.
       It assumes that different nodes and edges have the same dimension but exist in different vector spaces.

       The layer updates node features based on attention coefficients computed from
       node and edge embeddings.

       Parameters
       ----------
       in_channels_node : int=64
           Size of each input node feature.
       in_channels_edge_attr : int=5
           Size of each input edge attribute.
       in_channels_edge_type : int=4
           Size of each input edge type.
       node_emb_size : int=64
           Size of the node embedding.
       edge_attr_emb_size : int=64
           Size of the edge attribute embedding.
       edge_type_emb_size : int=64
           Size of the edge type embedding.
       out_channels : int=128
           Size of each output node feature.
       heads : int=3, optional
           Number of attention heads. (default: 3)
       concat : bool, optional
           If set to False, the multi-head attentions are averaged instead of concatenated. (default: True)

       """

    """
        1.consider 2 types of nodes: v, p
        2.consider 4 types of edges: v->v, v->p, p->v, p->p
        3.assume that different nodes have the same dimension, but different vector space.
        4.assume that different edges have the same dimension, but different vector space.
    """

    def __init__(self, in_channels_node=64, in_channels_edge_attr=5, in_channels_edge_type=4, node_emb_size=64,
                 edge_attr_emb_size=64, edge_type_emb_size=64, out_channels=128, heads=3, concat=True):
        super(HEATlayer, self).__init__()
        self.in_channels_node = in_channels_node
        self.in_channels_edge_attr = in_channels_edge_attr
        self.in_channels_edge_type = in_channels_edge_type
        self.node_emb_size = node_emb_size
        self.edge_attr_emb_size = edge_attr_emb_size
        self.edge_type_emb_size = edge_type_emb_size
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat

        # node feature embedding
        self.node_feat_emb = Linear(in_features=self.in_channels_node, out_features=self.node_emb_size, b_init=None, W_init='xavier_uniform')

        # edge attribute and edge type embedding
        self.edge_attr_emb = Linear(in_features=self.in_channels_edge_attr, out_features=self.edge_attr_emb_size, b_init=None, W_init='xavier_uniform')
        # self.edge_type_emb = Embedding(self.in_channels_edge_type, self.edge_type_emb_size)
        self.edge_type_emb = Linear(in_features=self.in_channels_edge_type, out_features=self.edge_type_emb_size, b_init=None, W_init='xavier_uniform')

        # attention layer
        self.attention_layer = Linear(in_features=2 * self.node_emb_size + self.edge_attr_emb_size + self.edge_type_emb_size,
                                      out_features=self.heads,  b_init=None, W_init='xavier_uniform')
        # update node feature
        self.update_node_emb = Linear(in_features=self.edge_attr_emb_size + self.node_emb_size, out_features=self.out_channels,  b_init=None, W_init='xavier_uniform')

        self.leaky_relu = LeakyReLU(negative_slope=0.2)
        self.softmax = tlx.nn.Softmax(axis=1)

    def embed_edges(self, edge_attrs, edge_types):
        """embed edge attributes and edge types respectively and combine them as the edge feature. """
        emb_edge_attr = self.leaky_relu(self.edge_attr_emb(edge_attrs))
        emb_edge_types = self.leaky_relu(self.edge_type_emb(edge_types))
        return emb_edge_attr, emb_edge_types

    def forward(self, x, edge_index, edge_attrs, edge_types):
        # node feature
        emb_node_feature = self.node_feat_emb(x)
        emb_node_feature = emb_node_feature.unsqueeze(axis=0).repeat(x.shape[0], 1, 1)

        emb_edge_attrs, emb_edge_types = self.embed_edges(edge_attrs, edge_types.float())
        emb_edge_features = tlx.concat([emb_edge_attrs, emb_edge_types], axis=-1)

        # eij+ [num_node, num_node, emb_edge_f_size + emb_node_size]
        emb_edge_features_cpu = tlx.convert_to_numpy(emb_edge_features)
        edge_index_cpu = tlx.convert_to_numpy(edge_index)
        size = [x.shape[0], x.shape[0], self.edge_attr_emb_size + self.edge_type_emb_size]

        update_emb_f = np.zeros(size)
        for i in range(edge_index_cpu.shape[1]):
            src_idx = edge_index_cpu[0, i]
            dst_idx = edge_index_cpu[1, i]
            update_emb_f[src_idx, dst_idx] = emb_edge_features_cpu[i]
        emb_edge_features = tlx.convert_to_tensor(update_emb_f)
        x_edge_f = tlx.concat((emb_node_feature, emb_node_feature, emb_edge_features), axis=-1).float()
        alpha = self.leaky_relu(self.attention_layer(x_edge_f))

        adj_np = np.zeros((alpha.shape[0], alpha.shape[1]))
        for i in range(edge_index.shape[1]):
            src_idx = edge_index[0, i]
            dst_idx = edge_index[1, i]
            adj_np[src_idx, dst_idx] = 1
        adj = tlx.convert_to_tensor(adj_np)
        adj = adj.unsqueeze(axis=2).repeat(1, 1, self.heads)
        score_nbrs = tlx.where(adj == 0.0, tlx.ones_like(alpha) * -10000, alpha)

        alpha = self.softmax(score_nbrs).unsqueeze(axis=3)
        alpha = alpha.repeat(1, 1, 1, self.out_channels)

        # update node feature
        emb_edge_attrs_cpu = tlx.convert_to_numpy(emb_edge_attrs)
        size = [x.shape[0], x.shape[0], self.edge_attr_emb_size]
        update_emb_f = np.zeros(size)

        for i in range(edge_index_cpu.shape[1]):
            src_idx = edge_index_cpu[0, i]
            dst_idx = edge_index_cpu[1, i]
            update_emb_f[src_idx, dst_idx] = emb_edge_attrs_cpu[i]

        emb_edge_attrs = tlx.convert_to_tensor(update_emb_f)

        x_attr_f = tlx.concat((emb_edge_attrs, emb_node_feature), axis=2).unsqueeze(axis=2).repeat(1, 1, self.heads, 1)
        out = self.leaky_relu(self.update_node_emb(x_attr_f.float()))
        out = tlx.reduce_sum(tlx.multiply(alpha, out), axis=1)

        if self.concat:
            out = tlx.reshape(out, (out.shape[0], -1))
        else:
            out = tlx.reduce_mean(out, axis=1)

        return out





