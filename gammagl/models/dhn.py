import tensorlayerx as tlx
from gammagl.layers.conv import DHNConv


class DHNModel(tlx.nn.Module):
    def __init__(self,
                 num_fea,
                 batch_size,
                 num_neighbor,
                 name=None):
        super().__init__(name=name)
        self.num_fea = num_fea
        self.batch_size = batch_size
        self.num_neighbor = num_neighbor
        self.dhn1 = DHNConv(num_fea, batch_size, num_neighbor)
        self.dhn2 = DHNConv(num_fea, batch_size, num_neighbor)
        self.lin1 = tlx.nn.Linear(in_features=4 * batch_size, out_features=batch_size, act=tlx.nn.ELU(),
                                  W_init="xavier_uniform")
        self.lin2 = tlx.nn.Linear(in_features=batch_size, out_features=1, act=tlx.nn.ELU(), W_init="xavier_uniform")

    def forward(self, n1, n2, label):
        n1_emb = self.dhn1(n1)
        n2_emb = self.dhn2(n2)

        pred = self.lin1(tlx.concat([n1_emb, n2_emb], axis=1))
        pred = self.lin2(pred)

        return pred
