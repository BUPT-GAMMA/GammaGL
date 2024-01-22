from tensorlayerx import nn
import tensorlayerx as tlx

class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance):
        super().__init__()
        self.max_path_distance = max_path_distance

        self.b = nn.Parameter(tlx.random_normal(tuple([self.max_path_distance])))

    def forward(self, x, paths):
        spatial_matrix = tlx.zeros((x.shape[0], x.shape[0])).to(next(self.parameters()).device)
        for src in paths:
            for dst in paths[src]:
                spatial_matrix[src][dst] = self.b[min(len(paths[src][dst]), self.max_path_distance) - 1]

        return spatial_matrix