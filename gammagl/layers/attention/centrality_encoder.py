from tensorlayerx import nn
from gammagl.utils import degree
import tensorlayerx as tlx


def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x

class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree, max_out_degree, node_dim):
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim
        self.z_in = tlx.nn.Parameter(tlx.ops.random_normal((max_in_degree, node_dim)))
        self.z_out = tlx.nn.Parameter(tlx.ops.random_normal((max_out_degree, node_dim)))

    def forward(self, x, edge_index):
        num_nodes = x.shape[0]

        in_degree = decrease_to_max_value(degree(index=edge_index[1], num_nodes=num_nodes).long(), self.max_in_degree)
        out_degree = decrease_to_max_value(degree(index=edge_index[0], num_nodes=num_nodes).long(), self.max_out_degree)

        x += self.z_in[in_degree] + self.z_out[out_degree]

        return x