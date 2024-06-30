import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing

type2idx = {
    'M': 0,
    'A': 1,
    # 'C': 2,
    # 'T': 3
}

NODE_TYPE = len(type2idx)
K_HOP = 2

NUM_FEA = (K_HOP + 2) * 4 + NODE_TYPE
NUM_NEIGHBOR = 5
BATCH_SIZE=32


class DHNConv(MessagePassing):
    def __init__(self):
        super().__init__()
        self.lin1 = tlx.nn.Linear(in_features=2*NUM_FEA, out_features=2*BATCH_SIZE, act=tlx.nn.ELU(), W_init="xavier_uniform")
        self.lin2 = tlx.nn.Linear(in_features=2*BATCH_SIZE+NUM_FEA, out_features=2*BATCH_SIZE, act=tlx.nn.ELU(), W_init="xavier_uniform")
        self.lin3 = tlx.nn.Linear(in_features=2*BATCH_SIZE, out_features=2*BATCH_SIZE, act=tlx.nn.ELU(), W_init="xavier_uniform")

    def aggregate(self, msg, edge_index, num_nodes=None, aggr=None):
        if len(msg.shape)==4:
            return tlx.reduce_mean(msg, axis=2)
        elif len(msg.shape)==3:
            return tlx.reduce_mean(msg, axis=1)

    def forward(self, fea):
        node = tlx.convert_to_tensor(fea[:, :NUM_FEA])

        # Extract neigh1 and neigh2
        neigh1 = tlx.convert_to_tensor(fea[:, NUM_FEA:NUM_FEA * (NUM_NEIGHBOR + 1)])
        neigh1 = tlx.reshape(neigh1, [-1, NUM_NEIGHBOR, NUM_FEA])

        neigh2 = tlx.convert_to_tensor(fea[:, NUM_FEA * (NUM_NEIGHBOR + 1):])
        neigh2 = tlx.reshape(neigh2, [-1, NUM_NEIGHBOR, NUM_NEIGHBOR, NUM_FEA])

        # aggregate on neigh2
        neigh2_agg = self.propagate(edge_index=None, x=neigh2)

        # connect the node to the aggregated neigh2
        tmp = tlx.concat(
            [neigh1, neigh2_agg],
            axis=2
        )

        # transform into a two-dimensional tensor
        flattened_tmp = tmp.view(2*BATCH_SIZE * NUM_NEIGHBOR, 2*NUM_FEA)
        flattened_output = self.lin1(flattened_tmp)
        # Reshape the output tensor to a three-dimensional shape
        tmp = flattened_output.view(2*BATCH_SIZE, NUM_NEIGHBOR, 2*BATCH_SIZE)

        # Aggregate node and tmp
        emb = tlx.concat(
            [
                node, self.propagate(edge_index=None,x=tmp)
            ],
            axis=1
        )

        # Through other linear layers
        emb = self.lin2(emb)
        emb = self.lin3(emb)

        return emb

    def message(self, x, edge_index, edge_weight=None):
        return x


class DHNModel(tlx.nn.Module):
    def __init__(self):
        super().__init__()
        self.dhn1 = DHNConv()
        self.dhn2 = DHNConv()
        self.lin1 = tlx.nn.Linear(in_features=4*BATCH_SIZE, out_features=BATCH_SIZE, act=tlx.nn.ELU(), W_init="xavier_uniform")
        self.lin2 = tlx.nn.Linear(in_features=BATCH_SIZE, out_features=1, act=tlx.nn.ELU(), W_init="xavier_uniform")

    def forward(self, n1, n2, label):
        n1_emb = self.dhn1(n1)
        n2_emb = self.dhn2(n2)

        pred = self.lin1(tlx.concat([n1_emb, n2_emb], axis=1))
        pred = self.lin2(pred)

        return pred
