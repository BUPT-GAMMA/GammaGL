import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.layers.conv import MessagePassing
from gammagl.utils import degree
from gammagl.layers.attention.sgformer_layer import TransConvLayer,GraphConvLayer

class SGFormerModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_class,
                 trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5,
                 gnn_num_layers=1, gnn_dropout=0.5, 
                 graph_weight=0.8, name=None):
        super().__init__(name=name)
        

        self.trans_convs = nn.ModuleList([
            TransConvLayer(feature_dim if i==0 else hidden_dim, 
                         hidden_dim, trans_num_heads, use_weight=True)
            for i in range(trans_num_layers)
        ])
        
 
        self.graph_convs = nn.ModuleList([
            GraphConvLayer(feature_dim if i==0 else hidden_dim,
                         hidden_dim, use_weight=True, use_init=False)
            for i in range(gnn_num_layers)
        ])
        
        self.fc = nn.Linear(in_features=hidden_dim, out_features=num_class)
        self.trans_dropout = trans_dropout
        self.gnn_dropout = gnn_dropout
        self.graph_weight = graph_weight
        self.activation = tlx.ReLU()

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):

        x1 = x
        for conv in self.trans_convs:
            x1 = conv(x1, x1)
            x1 = self.activation(x1)
            x1 = tlx.nn.Dropout(self.trans_dropout)(x1)
            

        x2 = x
        for conv in self.graph_convs:
            x2 = conv(x2, edge_index, x2, num_nodes)
            x2 = self.activation(x2)
            x2 = tlx.nn.Dropout(self.gnn_dropout)(x2)
            

        x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
        

        x = self.fc(x)
        return x 