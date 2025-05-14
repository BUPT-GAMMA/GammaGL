import tensorlayerx as tlx
from tensorlayerx import nn
from tensorlayerx.nn import Module, ModuleList, Embedding, Linear, LayerNorm
import numpy as np
from gammagl.layers.conv.egt_conv import Graph, VirtualNodes, EGTConv

NODE_FEATURES_OFFSET = 128
NUM_NODE_FEATURES = 9
EDGE_FEATURES_OFFSET = 8
NUM_EDGE_FEATURES = 3


class EGTModel(Module):
    r"""Edge-augmented Graph Transformer (EGT) model from the `"Global Self-Attention
    as a Replacement for Graph Convolution"
        <https://arxiv.org/abs/2108.03348>`_ paper.

    Parameters
    ----------
    node_width : int
        Dimensionality of node features.
    edge_width : int
        Dimensionality of edge features.
    num_heads : int
        Number of attention heads in multi-head attention mechanisms.
    model_height : int
        Number of layers in the model.
    node_mha_dropout : float
        Dropout rate applied to node multi-head attention layers.
    node_ffn_dropout : float
        Dropout rate applied to node feed-forward network layers.
    edge_mha_dropout : float
        Dropout rate applied to edge multi-head attention layers.
    edge_ffn_dropout : float
        Dropout rate applied to edge feed-forward network layers.
    attn_dropout : float
        Dropout rate applied to attention scores.
    attn_maskout : float
        Probability of masking out attention values during training.
    activation : str
        Activation function type.
    clip_logits_value : list
        Value range for clipping logits.
    node_ffn_multiplier : float
        Multiplier for determining hidden dimension in node feed-forward networks.
    edge_ffn_multiplier : float
        Multiplier for determining hidden dimension in edge feed-forward networks.
    scale_dot : bool
        Whether to scale dot products in attention mechanisms.
    scale_degree : bool
        Whether to scale attention by node degree.
    node_ended : bool
        Whether to update node features in the final layer.
    edge_ended : bool
        Whether to update edge features in the final layer.
    egt_simple : bool
        Whether to use a simplified version of the EGT layer.

    """

    def __init__(self,
                 node_width=128,
                 edge_width=32,
                 num_heads=8,
                 model_height=4,
                 node_mha_dropout=0.,
                 node_ffn_dropout=0.,
                 edge_mha_dropout=0.,
                 edge_ffn_dropout=0.,
                 attn_dropout=0.,
                 attn_maskout=0.,
                 activation='elu',
                 clip_logits_value=[-5, 5],
                 node_ffn_multiplier=2.,
                 edge_ffn_multiplier=2.,
                 scale_dot=True,
                 scale_degree=False,
                 node_ended=False,
                 edge_ended=False,
                 egt_simple=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.node_width = node_width
        self.edge_width = edge_width
        self.num_heads = num_heads
        self.model_height = model_height
        self.node_mha_dropout = node_mha_dropout
        self.node_ffn_dropout = node_ffn_dropout
        self.edge_mha_dropout = edge_mha_dropout
        self.edge_ffn_dropout = edge_ffn_dropout
        self.attn_dropout = attn_dropout
        self.attn_maskout = attn_maskout
        self.activation = activation
        self.clip_logits_value = clip_logits_value
        self.node_ffn_multiplier = node_ffn_multiplier
        self.edge_ffn_multiplier = edge_ffn_multiplier
        self.scale_dot = scale_dot
        self.scale_degree = scale_degree
        self.node_ended = node_ended
        self.edge_ended = edge_ended
        self.egt_simple = egt_simple

        self.layer_common_kwargs = dict(
            node_width=self.node_width,
            edge_width=self.edge_width,
            num_heads=self.num_heads,
            node_mha_dropout=self.node_mha_dropout,
            node_ffn_dropout=self.node_ffn_dropout,
            edge_mha_dropout=self.edge_mha_dropout,
            edge_ffn_dropout=self.edge_ffn_dropout,
            attn_dropout=self.attn_dropout,
            attn_maskout=self.attn_maskout,
            activation=self.activation,
            clip_logits_value=self.clip_logits_value,
            scale_dot=self.scale_dot,
            scale_degree=self.scale_degree,
            node_ffn_multiplier=self.node_ffn_multiplier,
            edge_ffn_multiplier=self.edge_ffn_multiplier,
        )

    def input_block(self, inputs):
        return Graph(inputs)

    def final_embedding(self, g):
        raise NotImplementedError

    def output_block(self, g):
        raise NotImplementedError

    def forward(self, inputs):
        raise NotImplementedError


class EGT(EGTModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.EGT_layers = ModuleList([EGTConv(**self.layer_common_kwargs,
                                              edge_update=(not self.egt_simple))
                                      for _ in range(self.model_height-1)])

        if (not self.node_ended) and (not self.edge_ended):
            pass
        elif not self.node_ended:
            self.EGT_layers.append(
                EGTConv(**self.layer_common_kwargs, node_update=False))
        elif not self.edge_ended:
            self.EGT_layers.append(
                EGTConv(**self.layer_common_kwargs, edge_update=False))
        else:
            self.EGT_layers.append(EGTConv(**self.layer_common_kwargs))

    def forward(self, inputs):
        g = self.input_block(inputs)

        for layer in self.EGT_layers:
            g = layer(g)

        g = self.final_embedding(g)

        outputs = self.output_block(g)
        return outputs


class EGT_MOL(EGT):
    def __init__(self,
                 upto_hop=16,
                 mlp_ratios=[1., 1.],
                 num_virtual_nodes=0,
                 svd_encodings=0,
                 output_dim=1,
                 **kwargs):
        super().__init__(node_ended=True, **kwargs)

        self.upto_hop = upto_hop
        self.mlp_ratios = mlp_ratios
        self.num_virtual_nodes = num_virtual_nodes
        self.svd_encodings = svd_encodings
        self.output_dim = output_dim

        self.nodef_embed = tlx.nn.Embedding(
            num_embeddings=NUM_NODE_FEATURES * NODE_FEATURES_OFFSET + 1,
            embedding_dim=self.node_width
        )
        if self.svd_encodings:
            self.svd_embed = Linear(
                out_features=self.node_width, in_features=self.svd_encodings * 2)

        self.dist_embed = Embedding(
            num_embeddings=self.upto_hop + 2, embedding_dim=self.edge_width)
        self.featm_embed = Embedding(num_embeddings=NUM_EDGE_FEATURES * EDGE_FEATURES_OFFSET + 1,
                                     embedding_dim=self.edge_width)

        if self.num_virtual_nodes > 0:
            self.vn_layer = VirtualNodes(self.node_width, self.edge_width,
                                         self.num_virtual_nodes)

        self.final_ln_h = LayerNorm(normalized_shape=self.node_width)
        mlp_dims = [self.node_width * max(self.num_virtual_nodes, 1)] \
            + [round(self.node_width * r) for r in self.mlp_ratios] \
            + [self.output_dim]
        self.mlp_layers = ModuleList([Linear(out_features=mlp_dims[i + 1], in_features=mlp_dims[i])
                                      for i in range(len(mlp_dims) - 1)])

        self.mlp_fn = getattr(tlx.ops, self.activation)

    def input_block(self, inputs):
        g = super().input_block(inputs)

        nodef = tlx.ops.cast(tlx.convert_to_tensor(
            g.node_features), dtype=tlx.int64)

        nodem = tlx.ops.cast(tlx.convert_to_tensor(
            g.node_mask), dtype=tlx.float32)

        dm0 = g.distance_matrix                     # (b,i,j)
        dm = tlx.ops.clip_by_value(
            tlx.convert_to_tensor(dm0),
            clip_value_min=np.iinfo(np.int64).min,
            clip_value_max=self.upto_hop + 1
        )
        dm = tlx.ops.cast(dm, dtype=tlx.int64)
        featm = tlx.ops.cast(tlx.convert_to_tensor(
            g.feature_matrix), dtype=tlx.int64)

        h = tlx.ops.reduce_sum(self.nodef_embed(
            nodef), axis=2)    # (b,i,w,h) -> (b,i,h)

        if self.svd_encodings:
            h = h + self.svd_embed(g.svd_encodings)

        e = self.dist_embed(dm) \
            + tlx.reduce_sum(self.featm_embed(featm),
                             axis=3)  # (b,i,j,f,e) -> (b,i,j,e)

        g.mask = (nodem[:, :, None, None] * nodem[:, None, :, None] - 1)*1e9
        g.h, g.e = h, e

        if self.num_virtual_nodes > 0:
            g = self.vn_layer(g)
        return g

    def final_embedding(self, g):
        h = g.h
        h = self.final_ln_h(h)
        if self.num_virtual_nodes > 0:
            h = tlx.reshape(h[:, :self.num_virtual_nodes], (h.shape[0], -1))
        else:
            nodem = tlx.expand_dims(
                tlx.cast(g.node_mask, dtype=tlx.float32), axis=-1)
            h = tlx.reduce_sum(h * nodem, axis=1) / \
                (tlx.reduce_sum(nodem, axis=1) + 1e-9)
        g.h = h
        return g

    def output_block(self, g):
        h = g.h
        h = self.mlp_layers[0](h)
        for layer in self.mlp_layers[1:]:
            h = layer(self.mlp_fn(h))
        return h


class EGT_PCQM4MV2(EGT_MOL):
    def __init__(self, **kwargs):
        super().__init__(output_dim=1, **kwargs)

    def output_block(self, g):
        h = super().output_block(g)
        return tlx.ops.squeeze(h, axis=-1)
