import tensorlayerx as tlx
from gammagl.layers.conv import AGNNConv


class AGNNModel(tlx.nn.Module):
    r"""The graph attention operator from the `"Attention-based Graph Neural Network for Semi-supervised Learning"
    <http://arxiv.org/abs/1803.03735>`_ paper


    Parameters
    ----------
    feature_dim: int
        Dimension of feature vector in original input.
    hidden_dim: intg
        Dimension of feature vector in AGNN.
    num_class: int
        Dimension of feature vector in forward output.
    n_att_layers: int
        Number of attention layers.
    dropout_rate: float
        Dropout rate.
    is_cora: bool,optional
        Whether the dateset is cora. There is a special operation on cora that cora dataset contains two agnn_conv layers.
    """

    def __init__(self,
                feature_dim,
                hidden_dim,
                num_class,
                n_att_layers,
                dropout_rate,
                is_cora = False,
                name = None):
        super().__init__(name = name)
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.n_att_layers = n_att_layers
        self.dropout_rate = dropout_rate
        self.dropout = tlx.layers.Dropout(self.dropout_rate)

        W_initor = tlx.initializers.XavierUniform()
        self.embedding_layer = tlx.nn.Linear(out_features = self.hidden_dim,
                                             W_init = W_initor,
                                            in_features = feature_dim)
        
        self.relu = tlx.nn.activation.ReLU()
        
        self.att_layers_list = []
        self.att_layers_list.append(AGNNConv(in_channels = self.hidden_dim,
                                            #Note:Only param of cora dataset in second agnn_conv layer doesn't have grad.
                                             require_grad = not(self.n_att_layers == 2 and is_cora)))
        for i in range(1, self.n_att_layers):
            self.att_layers_list.append(AGNNConv(in_channels = self.hidden_dim,))
        self.att_layers = tlx.nn.ModuleList(self.att_layers_list)
        
        self.output_layer = tlx.nn.Linear(out_features = self.num_class,
                                          W_init = W_initor,
                                          in_features = self.hidden_dim)

    def forward(self, x, edge_index, num_nodes):
        x = self.relu(self.embedding_layer(x))
        x = self.dropout(x)
        for i in range(len(self.att_layers)):
            x = self.att_layers[i](x, edge_index, num_nodes)
        x = self.output_layer(x)
        x = self.dropout(x)
        
        return x
