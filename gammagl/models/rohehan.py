# -*- coding: UTF-8 -*-
import tensorlayerx as tlx
from gammagl.layers.conv import RoheHANConv

class RoheHAN(tlx.nn.Module):
    """
    RoHeHAN: Robust Heterogeneous Graph Attention Network with Attention Purification

    This model implements a robust hierarchical attention mechanism specifically designed for 
    heterogeneous graphs. RoHeHAN applies two levels of attention:
    1. **Node-level attention**: Aggregates neighboring nodes based on metapaths, utilizing a robust attention mechanism 
       that incorporates edge transformations and attention purification.
    2. **Semantic-level attention**: Aggregates information across different metapaths (e.g., Paper-Author-Paper, 
       Paper-Field-Paper) to compute node representations. This step integrates knowledge from multiple edge types
       using a semantic attention mechanism.

    The model aims to counteract adversarial attacks by focusing on preserving relevant node connections while 
    filtering out unreliable or malicious neighbors via learned purification mechanisms.

    Parameters
    ----------
    metadata: tuple
        Metadata describing the heterogeneous graph, consisting of a list of node types and a list of edge types.
    in_channels: int or dict
        Input feature size for each node type. Can be a single integer for all node types or a dictionary mapping node 
        types to input feature sizes.
    hidden_size: int
        The size of hidden layers in the model.
    out_size: int
        The number of output classes for the final classification task.
    num_heads: list of int
        The number of attention heads used in each layer of the model.
    dropout_rate: float, optional
        Dropout probability for the attention mechanism (default: 0).
    settings: list of dict
        A list of dictionaries, where each dictionary contains settings for each layer and edge type. These settings 
        include parameters for edge transformation matrices and attention filtering.

    Methods
    -------
    forward:
        Performs a forward pass through the RoHeHAN layers and computes node representations for the target nodes.
    """
    def __init__(self, metadata, in_channels, hidden_size, out_size, num_heads, dropout_rate, settings):
        super(RoheHAN, self).__init__()
        self.layer_list = tlx.nn.ModuleList()

        # First layer: initialize with input feature dimensions
        self.layer_list.append(RoheHANConv(in_channels, hidden_size, metadata, num_heads[0], dropout_rate, settings[0]))

        # Subsequent layers: hidden size is multiplied by the number of attention heads from the previous layer
        for l in range(1, len(num_heads)):
            self.layer_list.append(
                RoheHANConv(hidden_size * num_heads[l-1], hidden_size, metadata, num_heads[l], dropout_rate, settings[l])
            )

        # Final classification layer
        self.predict = tlx.layers.Linear(in_features=hidden_size * num_heads[-1], out_features=out_size)

    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        for layer in self.layer_list:
            x_dict = layer(x_dict, edge_index_dict, num_nodes_dict)

        # Predict final class logits for each node type
        out = {node_type: self.predict(x) for node_type, x in x_dict.items()}
        return out
