import tensorlayerx as tlx
from gammagl.layers.conv import GCNConv
from gammagl.models import GraceModel

class LogReg(tlx.nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = tlx.nn.Linear(in_features=hid_dim, out_features=out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class Model(tlx.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, activation,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.grace = GraceModel(in_feat=in_dim, hid_feat=hidden_dim, out_feat=out_dim, num_layers=num_layers,
                                activation=activation, temp=tau)

    def get_embedding(self, x, edge_index, edge_attr):
        return self.grace.get_embeding(x, edge_index, edge_attr, num_nodes=x.shape[0]).detach()

    def forward(self, x_1, edge_index_1, edge_attr_1, x_2, edge_index_2, edge_attr_2):
        return self.grace(x_1, edge_index_1, edge_attr_1, x_1.shape[0], x_2, edge_index_2, edge_attr_2, x_2.shape[0])
