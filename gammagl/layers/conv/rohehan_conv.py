# -*- coding: UTF-8 -*-
import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax
from gammagl.ops.functional import _unique_impl
import numpy as np


class RoheGATConv(MessagePassing):
    """
    RoHeGATConv Layer (Graph Attention Layer with Attention Purification)

    This layer extends traditional graph attention mechanisms with a defense mechanism against 
    adversarial attacks on heterogeneous graphs. It computes attention scores between nodes and
    integrates an edge transformation matrix, leveraging metapath-based transiting probabilities 
    to filter out malicious or unreliable neighbors based on topology and feature information.

    The main idea is to counteract the **perturbation enlargement effect** and prevent adversarial hubs 
    from dominating attention mechanisms by pruning such nodes with a learned purifier.

    Parameters
    ----------
    in_channels: int
        The number of input features for each node.
    out_channels: int
        The number of output features for each node.
    num_heads: int
        The number of attention heads.
    dropout_rate: float, optional
        The dropout probability (default: 0.0).
    negative_slope: float, optional
        The negative slope for the LeakyReLU activation (default: 0.2).
    residual: bool, optional
        Whether to include a residual connection (default: False).
    settings: dict, optional
        Settings for edge transformation, including parameters such as the number of edges to retain
        and the transformation matrix.
    """
    def __init__(self, in_channels, out_channels, num_heads, dropout_rate=0., negative_slope=0.2, residual=False, settings=None):
        super(RoheGATConv, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.settings = settings or {'K': 10, 'P': 0.6, 'tau': 0.1, 'Flag': "None", 'T': 1, 'TransM': None}
        self.feat_drop = tlx.layers.Dropout(dropout_rate)
        self.attn_drop = tlx.layers.Dropout(dropout_rate)
        self.leaky_relu = tlx.layers.LeakyReLU(negative_slope)

        # Linear transformation for node features
        self.fc = tlx.layers.Linear(in_features=in_channels, out_features=out_channels * num_heads, b_init=None)

        # Residual connection setup
        if residual:
            self.res_fc = tlx.layers.Linear(in_features=in_channels, out_features=num_heads * out_channels, b_init=None)
        else:
            self.res_fc = None

    def forward(self, x, edge_index, num_nodes):
        edge_trans_values = self.settings['TransM']  # Transformation matrix for the edges
        T = self.settings['T']  # Number of edges to keep per node

        # Apply dropout to input node features
        x = self.feat_drop(x)

        # Apply linear transformation to node features
        x = self.fc(x)
        x = tlx.reshape(x, (-1, self.num_heads, self.out_channels))
        edge_index = tlx.convert_to_tensor(edge_index, dtype=tlx.int64)

        # Gather source and destination node features
        node_src = edge_index[0, :]
        node_dst = edge_index[1, :]
        feat_src = tlx.gather(x, node_src)
        feat_dst = tlx.gather(x, node_dst)

        # Compute attention scores using dot product
        e = tlx.reduce_sum(feat_src * feat_dst, axis=-1)

        # Apply LeakyReLU to attention scores
        e = self.leaky_relu(e)

        # Apply transformation matrix to attention scores
        edge_trans_values = edge_trans_values[:, np.newaxis]
        edge_trans_values = np.repeat(edge_trans_values, self.num_heads, axis=1)
        edge_trans_values = tlx.convert_to_tensor(edge_trans_values, dtype=tlx.float32)

        attn = e * edge_trans_values

        # Top-T selection for edges based on attention scores
        node_dst_np = tlx.convert_to_numpy(node_dst)
        attn_np = tlx.convert_to_numpy(attn)

        # Create a mask to retain only the top-T edges for each node
        node_edge_dict = {}
        for idx, node in enumerate(node_dst_np):
            if node not in node_edge_dict:
                node_edge_dict[node] = []
            node_edge_dict[node].append(idx)

        mask = np.zeros((attn.shape[0], self.num_heads), dtype=np.bool_)
        for node, edge_indices in node_edge_dict.items():
            edge_indices = np.array(edge_indices)
            for h in range(self.num_heads):
                attn_values = attn_np[edge_indices, h]
                if len(attn_values) <= T:
                    selected_edges = edge_indices
                else:
                    topk_indices = np.argsort(-attn_values)[:T]
                    selected_edges = edge_indices[topk_indices]
                mask[selected_edges, h] = True

        # Apply mask to attention values
        mask = tlx.convert_to_tensor(mask, dtype=tlx.bool)
        very_negative_value = -9e15
        attn = tlx.where(mask, attn, very_negative_value)

        # Apply softmax normalization for attention values
        alpha = segment_softmax(attn, node_dst, num_nodes)

        # Apply attention to node features
        out = self.propagate(x, edge_index, num_nodes=num_nodes, edge_weight=alpha)

        # Reshape the output and apply residual connection if available
        out = tlx.reshape(out, (-1, self.num_heads * self.out_channels))
        if self.res_fc:
            res = self.res_fc(x)
            out += res

        return out


class SemanticAttention(tlx.nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = tlx.nn.Sequential(
            tlx.layers.Linear(in_features=in_size, out_features=hidden_size),
            tlx.layers.Tanh(),
            tlx.layers.Linear(in_features=hidden_size, out_features=1, b_init=None)
        )

    def forward(self, z):
        w = tlx.reduce_mean(self.project(z), axis=1)
        beta = tlx.softmax(w, axis=0)
        beta = tlx.expand_dims(beta, axis=-1)
        return tlx.reduce_sum(beta * z, axis=0)


class RoheHANConv(tlx.nn.Module):
    """
    RoHeHANConv: Heterogeneous Graph Attention Layer with Attention Purification

    This layer implements a robust hierarchical attention mechanism for heterogeneous graphs, combining 
    both node-level and semantic-level attention. It mitigates the impact of adversarial attacks by using 
    attention purification, which selectively prunes unreliable or malicious neighbors.

    The node-level attention aggregates neighbors based on metapaths, while the semantic-level attention 
    fuses information from multiple metapaths. 

    Parameters
    ----------
    in_channels: int or dict
        The number of input features for each node type. Can be a single integer for all node types 
        or a dictionary mapping node types to input feature sizes.
    out_channels: int
        The number of output features for each node.
    metadata: tuple
        Metadata describing the heterogeneous graph, consisting of a list of node types and a list of edge types.
    num_heads: int
        The number of attention heads.
    dropout_rate: float, optional
        Dropout probability for the attention mechanism (default: 0.5).
    settings: dict
        Settings for each edge type, including transformation matrices and attention parameters.
    """
    def __init__(self, in_channels, out_channels, metadata, num_heads, dropout_rate, settings):
        super(RoheHANConv, self).__init__()
        # Initialize GAT layers for each edge type in the metadata
        if isinstance(in_channels, int):
            self.gat_layers = tlx.nn.ModuleDict({
                '__'.join(edge_type): RoheGATConv(in_channels, out_channels, num_heads, dropout_rate, settings=settings[edge_type])
                for edge_type in metadata[1]
            })
        else:
            self.gat_layers = tlx.nn.ModuleDict({
                '__'.join(edge_type): RoheGATConv(in_channels[edge_type[0]], out_channels, num_heads, dropout_rate, settings=settings[edge_type])
                for edge_type in metadata[1]
            })

        # Semantic attention layer for aggregating across different metapaths
        self.semantic_attention = SemanticAttention(out_channels * num_heads)
        self.metadata = metadata

    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        out_dict = {key: [] for key in x_dict.keys()}

        # Apply GATConv for each edge type
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if len(edge_index) > 0 and src_type in x_dict:
                out = self.gat_layers['__'.join(edge_type)](x_dict[src_type], edge_index, num_nodes_dict[dst_type])
                out_dict[dst_type].append(out)

        # Apply semantic attention across different edge types for each node type
        for node_type, outs in out_dict.items():
            if outs:
                out_dict[node_type] = self.semantic_attention(tlx.stack(outs))
            else:
                # If no outputs exist for this node type, initialize a zero tensor
                out_dict[node_type] = tlx.zeros((num_nodes_dict[node_type], self.gat_layers[list(self.gat_layers.keys())[0]].out_channels * self.gat_layers[list(self.gat_layers.keys())[0]].num_heads))

        return out_dict
