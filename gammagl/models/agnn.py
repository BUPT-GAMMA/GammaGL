import tensorlayerx as tlx
from gammagl.layers.conv import AGNNConv


def segment_log_softmax(weight, segment_ids, num_nodes):
    max_values = tlx.ops.unsorted_segment_max(weight, segment_ids, num_segments = num_nodes)
    gathered_max_values = tlx.ops.gather(max_values, segment_ids)
    weight = weight - gathered_max_values
    exp_weight = tlx.ops.exp(weight)

    sum_weights = tlx.ops.unsorted_segment_sum(exp_weight, segment_ids, num_segments = num_nodes)
    sum_weights = tlx.ops.gather(sum_weights, segment_ids)
    softmax_weight = tlx.ops.divide(exp_weight, sum_weights)

    return softmax_weight

class AGNNModel(tlx.nn.Module):

    def __init__(self,
                feature_dim,
                hidden_dim,
                num_class,
                n_att_layers,
                dropout_rate,
                edge_index,
                num_nodes,
                is_cora = False,
                name = None):
        super().__init__(name = name)
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.n_att_layers = n_att_layers
        self.dropout_rate = dropout_rate
        self.dropout = tlx.layers.Dropout(1 - self.dropout_rate)

        W_initor = tlx.initializers.XavierUniform()
        self.embedding_layer = tlx.layers.Dense(n_units = self.hidden_dim,
                                                W_init = W_initor,
                                                in_channels = feature_dim)
        
        self.relu = tlx.nn.activation.ReLU()
        
        self.att_layers_list = []
        self.att_layers_list.append(AGNNConv(in_channels  = self.hidden_dim,
                                        out_channels =  self.hidden_dim,
                                        edge_index = edge_index,
                                        num_nodes = num_nodes,
                                        require_grad = not(self.n_att_layers == 2 and is_cora)))
        for i in range(1, self.n_att_layers):
            self.att_layers_list.append(AGNNConv(in_channels  = self.hidden_dim,
                                            out_channels =  self.hidden_dim,
                                            edge_index = edge_index,
                                            num_nodes = num_nodes))
        self.att_layers = tlx.nn.SequentialLayer(self.att_layers_list)
        
        self.output_layer = tlx.layers.Dense(n_units = self.num_class,
                                             W_init = W_initor,
                                             in_channels = self.hidden_dim)

    def forward(self, x):
        x = self.relu(self.embedding_layer(x))
        x = self.dropout(x)

        x = self.att_layers(x)
            
        x = self.output_layer(x)
        x = self.dropout(x)
        
        return x
        

        
