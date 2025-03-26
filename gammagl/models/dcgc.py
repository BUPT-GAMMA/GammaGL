import tensorlayerx as tlx
from tensorlayerx.nn import Module
import tensorlayerx.nn as nn

class EdgeWeight(Module):
    def __init__(self, out_channels, base_model, dropout):
        super(EdgeWeight, self).__init__()
        self.base_model = base_model
        self.name = 'EW'
        self.extractor = nn.Sequential(tlx.nn.Linear(in_features=out_channels*2, out_features=out_channels*4),
                                       tlx.nn.ReLU(),
                                       tlx.nn.Dropout(dropout),
                                       tlx.nn.Linear(in_features=out_channels*4, out_features=1))

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        
        if edge_weight is None:
            edge_weight = self.get_weight(x, edge_index, num_nodes)
        logits = self.base_model(x, edge_index, edge_weight, num_nodes)
        return logits

    def get_weight(self, x, edge_index, num_nodes):
        
        emb = self.base_model(x, edge_index, edge_weight=None, num_nodes=num_nodes)
        col, row = edge_index
        f1, f2 = tlx.gather(emb, col), tlx.gather(emb, row)
        f12 = tlx.concat([f1, f2], axis=-1)
        edge_weight = self.extractor(f12)
        
        return tlx.relu(edge_weight)


class TemperatureScaling(Module):
    def __init__(self, base_model):
        super(TemperatureScaling, self).__init__()
        self.base_model = base_model
        self.name = "TS"
        self.temperature = tlx.nn.Parameter(tlx.ones((1,)))

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        logits = self.base_model(x, edge_index, edge_weight, num_nodes)
        temperature = tlx.tile(self.temperature, [logits.shape[0], logits.shape[1]])
        return tlx.ops.multiply(logits, temperature)

    def reset_parameters(self):
        self.temperature.assign(tlx.ones((1,)))
