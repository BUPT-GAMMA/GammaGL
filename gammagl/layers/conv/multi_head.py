import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax

class MultiHead(MessagePassing):
    def __init__(self, in_features, out_features, n_heads,num_nodes):
        super().__init__()
        self.heads=n_heads
        self.num_nodes=num_nodes
        self.out_channels=out_features
        self.linear = tlx.layers.Linear(out_features=out_features* n_heads,
                                        in_features=in_features)

        init = tlx.initializers.RandomNormal()
        self.att_src = init(shape=(1, n_heads, out_features), dtype=tlx.float32)
        self.att_dst = init(shape=(1, n_heads, out_features), dtype=tlx.float32)

        self.leaky_relu = tlx.layers.LeakyReLU(0.2)
        self.dropout = tlx.layers.Dropout()

    def message(self, x, edge_index):
        node_src = edge_index[0, :]
        node_dst = edge_index[1, :]
        weight_src = tlx.gather(tlx.reduce_sum(x * self.att_src, -1), node_src)
        weight_dst = tlx.gather(tlx.reduce_sum(x * self.att_dst, -1), node_dst)
        weight = self.leaky_relu(weight_src + weight_dst)

        alpha = self.dropout(segment_softmax(weight, node_dst, self.num_nodes))
        x = tlx.gather(x, node_src) * tlx.expand_dims(alpha, -1)
        return x
    
    def forward(self, x, edge_index):
        x = tlx.reshape(self.linear(x), shape=(-1,self.heads, self.out_channels))
        x = self.propagate(x, edge_index, num_nodes=self.num_nodes)
        x=tlx.ops.reduce_mean(x,axis=1)

        return x