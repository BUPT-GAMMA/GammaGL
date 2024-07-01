import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing


class DHNConv(MessagePassing):
    def __init__(self,
                 num_fea,
                 batch_size,
                 num_neighbor):
        super().__init__()
        self.num_fea = num_fea
        self.batch_size = batch_size
        self.num_neighbor = num_neighbor
        self.lin1 = tlx.nn.Linear(in_features=2 * num_fea, out_features=2 * batch_size, act=tlx.nn.ELU(),
                                  W_init="xavier_uniform")
        self.lin2 = tlx.nn.Linear(in_features=2 * batch_size + num_fea, out_features=2 * batch_size, act=tlx.nn.ELU(),
                                  W_init="xavier_uniform")
        self.lin3 = tlx.nn.Linear(in_features=2 * batch_size, out_features=2 * batch_size, act=tlx.nn.ELU(),
                                  W_init="xavier_uniform")

    def aggregate(self, msg, edge_index, num_nodes=None, aggr=None):
        if len(msg.shape) == 4:
            return tlx.reduce_mean(msg, axis=2)
        elif len(msg.shape) == 3:
            return tlx.reduce_mean(msg, axis=1)

    def forward(self, fea):
        node = tlx.convert_to_tensor(fea[:, :self.num_fea])

        # Extract neigh1 and neigh2
        neigh1 = tlx.convert_to_tensor(fea[:, self.num_fea:self.num_fea * (self.num_neighbor + 1)])
        neigh1 = tlx.reshape(neigh1, [-1, self.num_neighbor, self.num_fea])

        neigh2 = tlx.convert_to_tensor(fea[:, self.num_fea * (self.num_neighbor + 1):])
        neigh2 = tlx.reshape(neigh2, [-1, self.num_neighbor, self.num_neighbor, self.num_fea])

        # aggregate on neigh2
        neigh2_agg = self.propagate(edge_index=None, x=neigh2)

        # connect the node to the aggregated neigh2
        tmp = tlx.concat(
            [neigh1, neigh2_agg],
            axis=2
        )

        # transform into a two-dimensional tensor
        flattened_tmp = tmp.view(2 * self.batch_size * self.num_neighbor, 2 * self.num_fea)
        flattened_output = self.lin1(flattened_tmp)
        # Reshape the output tensor to a three-dimensional shape
        tmp = flattened_output.view(2 * self.batch_size, self.num_neighbor, 2 * self.batch_size)

        # Aggregate node and tmp
        emb = tlx.concat(
            [
                node, self.propagate(edge_index=None, x=tmp)
            ],
            axis=1
        )

        # Through other linear layers
        emb = self.lin2(emb)
        emb = self.lin3(emb)

        return emb

    def message(self, x, edge_index, edge_weight=None):
        return x
