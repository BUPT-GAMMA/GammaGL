import tensorlayerx as tlx
from gammagl.layers.conv import CompConv
class CompGCN(tlx.nn.Module):
    """Composition-based Multi-Relational graph convoluation nerworks"""
    def __init__(self, feature_dim, hidden_dim, num_class,op, num_relations,num_entity, name=None):
        super().__init__(name=name)
        self.op = op
        self.conv1 = CompConv(feature_dim, hidden_dim, num_relations,op)
        self.conv2 = CompConv(hidden_dim, num_class,num_relations,op)
        self.relu = tlx.ReLU()
        self.init_input = tlx.random_normal(shape=(num_entity, feature_dim), dtype=tlx.float32)
        self.ref_input = tlx.random_normal(shape=(num_relations + 1, 32), dtype=tlx.float32)
    def forward(self, edge_index, edge_type):
        x,r = self.conv1(self.init_input, edge_index, edge_type, self.ref_input)
        x = self.relu(x)
        x,r = self.conv2(x, edge_index, edge_type, r)
        return x