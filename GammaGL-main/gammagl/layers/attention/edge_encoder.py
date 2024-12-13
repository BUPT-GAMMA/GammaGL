from tensorlayerx import nn
from gammagl.utils import degree
import tensorlayerx as tlx
import numpy as np


def dot_product(x1, x2):
    return tlx.reduce_sum((x1 * x2), axis=1)

class EdgeEncoding(nn.Module):
    def __init__(self, edge_dim, max_path_distance):
        super().__init__()
        self.edge_dim = edge_dim
        self.max_path_distance = max_path_distance

        self.edge_vector = nn.Parameter(tlx.random_normal((self.max_path_distance, self.edge_dim)))

    def forward(self, x, edge_attr, edge_paths):
        cij = tlx.zeros((x.shape[0], x.shape[0]))
        cij = tlx.convert_to_numpy(cij)

        for src in edge_paths:
            for dst in edge_paths[src]:
                path_ij = edge_paths[src][dst][:self.max_path_distance]
                weight_inds = [i for i in range(len(path_ij))]
                if path_ij == []:
                    continue
                cij[src][dst] = tlx.reduce_mean(dot_product(tlx.gather(self.edge_vector, weight_inds), tlx.gather(edge_attr, path_ij)))

        cij_no_nan = np.nan_to_num(cij)
        cij = tlx.convert_to_tensor(cij_no_nan)
        return cij