import tensorlayerx as tlx
from gammagl.models import GINModel, GCNModel, GATModel, GraphSAGE_Full_Model, MLP
from gammagl.layers.pool.glob import global_sum_pool

class DFADModel(tlx.nn.Module):
    def __init__(self, model_name, feature_dim, hidden_dim, num_classes, num_layers, drop_rate):
        super(DFADModel, self).__init__()
        if model_name == "gcn":
            self.gnn = GCNModel(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_class=hidden_dim, 
                num_layers=num_layers
            )
        elif model_name == "gin":
            self.gnn = GINModel(
                in_channels=feature_dim,
                hidden_channels=hidden_dim,
                out_channels=num_classes,
                num_layers=num_layers
            )
        elif model_name == "gat":
            self.gnn = GATModel(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_class=hidden_dim,
                heads=3,
                drop_rate=drop_rate,
                num_layers=num_layers
            )
        elif model_name == "graphsage":
            self.gnn = GraphSAGE_Full_Model(
                in_feats=feature_dim,
                n_hidden=hidden_dim,
                n_classes=hidden_dim,
                n_layers=num_layers,
                activation=tlx.nn.ReLU(),
                dropout=drop_rate,
                aggregator_type="mean"
            )
        else:
            raise NameError("model name error")

        self.model_name = model_name
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.mlp = MLP([hidden_dim, hidden_dim, num_classes])



    def forward(self, x, edge_index, num_nodes, batch):
        if self.model_name == "gcn":
            logits = self.gnn(x, edge_index, None, num_nodes)
        elif self.model_name == "gin":
            logits = self.gnn(x, edge_index, batch)
        elif self.model_name == "gat":
            logits = self.gnn(x, edge_index, num_nodes)
        elif self.model_name == "graphsage":
            logits = self.gnn(x, edge_index)
        else:
            raise NameError("model name error")

        if self.model_name != "gin":
            logits = global_sum_pool(logits, batch)
            return self.mlp(logits)
        else:
            return logits

class DFADGenerator(tlx.nn.Module):
    def __init__(self, conv_dims, z_dim, num_vertices, num_features, dropout):
        super(DFADGenerator, self).__init__()

        self.num_vertices = num_vertices
        self.num_features = num_features

        layers = []
        for c0, c1 in zip([z_dim]+conv_dims[:-1], conv_dims):
            layers.append(tlx.nn.Linear(in_features=c0, out_features=c1))
            layers.append(tlx.nn.Tanh())  
            layers.append(tlx.nn.Dropout(p=dropout))

        self.layer_list = tlx.nn.Sequential(*layers)
        self.nodes_layer = tlx.layers.Linear(in_features=conv_dims[-1], out_features=num_vertices * num_features)
        
    def forward(self, x):
        output = self.layer_list(x)
        nodes_logits = self.nodes_layer(output)
        nodes_logits = tlx.reshape(nodes_logits, [-1, self.num_vertices, self.num_features])
        
        adj = tlx.bmm(nodes_logits,nodes_logits.permute(0,2,1))
        adj = tlx.cast(adj, tlx.int64)
        return adj, nodes_logits