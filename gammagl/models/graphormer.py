from gammagl.data import Graph
import tensorlayerx as tlx
from tensorlayerx import nn

from ..utils import shortest_path_distance, batched_shortest_path_distance
from ..layers.attention import GraphormerLayer, CentralityEncoding, SpatialEncoding

class Graphormer(nn.Module):
    
    r"""The graph transformer model from the `"Do Transformers Really Perform Bad for 
    Graph Representation?" <https://arxiv.org/abs/2106.05234>`_ paper.

        Parameters
        ----------
        num_layers: int
            number of Graphormer layers.
        input_node_dim: int
            input dimension of node features.
        node_dim: int
            hidden dimensions of node features.
        input_edge_dim: int
            input dimension of edge features.
        edge_dim: int
            hidden dimensions of edge features.
        output_dim: int
            number of output node features.
        n_heads: int
            number of attention heads.
        max_in_degree: int
            max in degree of nodes.
        max_out_degree: int
            max out degree of nodes.
        max_path_distance: int
            max pairwise distance between two nodes.

    """
    
    def __init__(self,
                 num_layers: int,
                 input_node_dim: int,
                 node_dim: int,
                 input_edge_dim: int,
                 edge_dim: int,
                 output_dim: int,
                 n_heads: int,
                 max_in_degree: int,
                 max_out_degree: int,
                 max_path_distance: int,
                 name=None):
        super().__init__(name=name)

        self.num_layers = num_layers
        self.input_node_dim = input_node_dim
        self.node_dim = node_dim
        self.input_edge_dim = input_edge_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.max_path_distance = max_path_distance
        
        self.node_in_lin = nn.Linear(in_features=self.input_node_dim, out_features=self.node_dim)
        self.edge_in_lin = nn.Linear(in_features=self.input_edge_dim, out_features=self.edge_dim)

        self.centrality_encoding = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            node_dim=self.node_dim
        )

        self.spatial_encoding = SpatialEncoding(
            max_path_distance=max_path_distance,
        )

        self.layers = tlx.nn.ModuleList([
            GraphormerLayer(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                n_heads=self.n_heads,
                max_path_distance=self.max_path_distance) for _ in range(self.num_layers)
        ])

        self.node_out_lin = nn.Linear(in_features=self.node_dim, out_features=self.output_dim)

    def forward(self, data):
        x = tlx.cast(data.x, dtype=tlx.float32)
        edge_index = tlx.cast(data.edge_index, dtype=tlx.int64)
        edge_attr = tlx.cast(data.edge_attr, dtype=tlx.float32)

        if isinstance(data, Graph):
            ptr = None
            node_paths, edge_paths = shortest_path_distance(data)
        else:
            ptr = data.ptr
            node_paths, edge_paths = batched_shortest_path_distance(data)

        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)

        x = self.centrality_encoding(x, edge_index)
        b = self.spatial_encoding(x, node_paths)

        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths, ptr)

        x = self.node_out_lin(x)

        return x
