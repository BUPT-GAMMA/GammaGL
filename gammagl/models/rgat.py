import tensorlayerx as tlx
from gammagl.layers.conv import RGATConv
#from torch_geometric.nn import RGATConv


class RGAT(tlx.nn.Module):
    "Relational Graph Attention Networks"
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_relations, attention_mechanism, attention_mode,
                 dim, heads, concat, negative_slope, bias):
        super().__init__()

        self.conv1 = RGATConv(in_channels, hidden_channels, num_relations, 
                              attention_mechanism, attention_mode, heads,
                              dim, concat, negative_slope, bias)

        #self.conv1 = RGATConv(in_channels, hidden_channels, num_relations,)

        self.conv2 = RGATConv(hidden_channels, out_channels, num_relations,
                              attention_mechanism, attention_mode, heads,
                              dim, concat, negative_slope, bias)

        #self.conv2 = RGATConv(hidden_channels, hidden_channels, num_relations)
        self.lin = tlx.nn.Linear(hidden_channels, out_channels)
        self.relu = tlx.ReLU()
        #self.log_softmax = tlx.ops.log(tlx.nn.Softmax())
        self.log_softmax = tlx.nn.Softmax()
    def forward(self, edge_index, edge_type):
        x = self.relu(self.conv1(None, edge_index, edge_type))
        x = self.relu(self.conv2(x, edge_index, edge_type))
        #x = self.lin(x)
        return  self.log_softmax(x,dim=-1)
