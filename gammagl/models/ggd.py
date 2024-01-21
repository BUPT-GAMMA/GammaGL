import tensorlayerx as tlx
import tensorlayerx.nn as nn
import copy
from gammagl.layers.conv import GCNConv
from gammagl.utils import to_scipy_sparse_matrix

class GGDModel(nn.Module):
    def __init__(self, n_in, n_h, nb_classes):
        super(GGDModel, self).__init__()
        self.gcn = GCNConv(n_in, n_h, norm='none')
        self.act = tlx.nn.PRelu()
        self.lin = nn.Linear(out_features=n_h, in_features=n_h, W_init='xavier_uniform',
                             b_init=tlx.initializers.zeros())

    def forward(self, seq1, seq2, edge_index, edge_weight, num_nodes):
        h_1 = self.gcn(seq1, edge_index, edge_weight, num_nodes)
        h_1 = self.act(h_1)
        h_2 = self.gcn(seq2, edge_index, edge_weight, num_nodes)
        h_2 = self.act(h_2)
        sc_1 = tlx.expand_dims(tlx.reduce_sum(self.lin(h_1), 1), 0)
        sc_2 = tlx.expand_dims(tlx.reduce_sum(self.lin(h_2), 1), 0)

        logits = tlx.concat((sc_1, sc_2), 1)
        return logits

    # Detach the return variables
    def embed(self, seq, edge_index, edge_weight):
        h_1 = self.gcn(seq, edge_index, edge_weight)
        h_2 = tlx.squeeze(copy.copy(h_1), axis=0)
        adj = to_scipy_sparse_matrix(edge_index, edge_weight)
        for i in range(5):
            h_2 = tlx.convert_to_tensor(adj.A) @ h_2

        h_2 = tlx.expand_dims(h_2, axis=0)

        return tlx.detach(h_1), tlx.detach(h_2)
