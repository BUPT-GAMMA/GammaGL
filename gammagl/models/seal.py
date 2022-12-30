import tensorlayerx as tlx
from gammagl.layers.conv import GCNConv, SAGEConv
from gammagl.layers.pool import global_sort_pool
import math


class DGCNN(tlx.nn.Module):
    r"""
    DGCNN proposed in `An End-to-End Deep Learning Architecture for Graph Classiﬁcation`_.

    .. _An End-to-End Deep Learning Architecture for Graph Classiﬁcation:
        https://dl.acm.org/doi/pdf/10.5555/3504035.3504579

    Parameters
    ----------
        feature_dim (int): input feature dimension
        hidden_dim (int): hidden dimension
        num_layers (int): number of layers
        gcn_type (str): convolution layer type
        k (int or float): The number of nodes to hold for each graph in SortPooling
        train_dataset (dataset): train dataset to extract minimum number of nodes to generate k
        dropout (float): dropout rate
        name (str): model name
    """

    def __init__(self, feature_dim,
                 hidden_dim,
                 num_layers,
                 gcn_type = 'gcn',
                 k = 0.6,
                 train_dataset=None,
                 dropout = 0.5,
                 name=None):
        if gcn_type == 'gcn':
            GNN = GCNConv
        else:
            GNN = SAGEConv

        super().__init__(name=name)

        if k < 1:  # Transform percentile to number.
            if train_dataset is None:
                k = 30
            else:
                num_nodes = sorted([g.num_nodes for g in train_dataset])
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        self.k = int(k)

        self.convs = tlx.nn.ModuleList()
        self.convs.append(GNN(feature_dim, hidden_dim))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_dim, hidden_dim))
        self.convs.append(GNN(hidden_dim, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_dim * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = tlx.nn.Conv1d(
            out_channels=conv1d_channels[0],
            kernel_size=conv1d_kws[0],
            stride=conv1d_kws[0],
            act=tlx.nn.ReLU,
            padding='VALID'
        )
        self.maxpool1d = tlx.nn.MaxPool1d(2, 2, 'VALID')
        self.conv2 = tlx.nn.Conv1d(
            out_channels=conv1d_channels[1],
            kernel_size=conv1d_kws[1],
            act=tlx.nn.ReLU,
            padding='VALID'
        )
        self.lin1 = tlx.nn.Linear(out_features=128, act=tlx.nn.ReLU)
        self.drop = tlx.nn.Dropout(p=dropout)
        self.lin2 = tlx.nn.Linear(out_features=1)

    def forward(self, x, edge_index, batch):
        xs = [x]
        for conv in self.convs:
            xs += [tlx.tanh(conv(xs[-1], edge_index))]
        x = tlx.concat(xs[1:], axis=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = tlx.expand_dims(x, -1) # [num_graphs, k * hidden, 1]
        x = self.conv1(x)
        x = self.maxpool1d(x)
        x = self.conv2(x)
        x = tlx.reshape(x, (x.shape[0], -1))  # [num_graphs, dense_dim]

        # MLP.
        x = self.lin1(x)
        x = self.drop(x)
        x = self.lin2(x)

        return x