import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.layers.conv import GCNConv
import torch
from torch_geometric.nn import GCNConv as tGCNConv, GATConv
import torch.nn as tnn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree



class GCNModel(tlx.nn.Module):
    r"""Graph Convolutional Network proposed in `"Semi-supervised
        Classification with Graph Convolutional Networks"
        <https://arxiv.org/abs/1609.02907>`_ paper.
        
        Parameters
        ----------
        feature_dim: int
            input feature dimension.
        hidden_dim: int
            hidden dimension.
        num_class: int
            number of classes.
        drop_rate: float, optional
            dropout rate.
        num_layers: int, optional
            number of layers.
        norm: str, optional
            apply the normalizer.
        name: str, optional
            model name.

    """

    def __init__(self, feature_dim,
                 hidden_dim,
                 num_class,
                 drop_rate = 0.2,
                 num_layers = 2,
                 norm = 'both',
                 name=None):
        super().__init__(name=name)
        self.num_layers = num_layers
        if num_layers == 1:
            self.conv = nn.ModuleList([GCNConv(feature_dim, num_class, norm = norm)])
        else:
            self.conv = nn.ModuleList([GCNConv(feature_dim, hidden_dim, norm = norm)])
            for _ in range(1, num_layers - 1):
                self.conv.append(GCNConv(hidden_dim, hidden_dim, norm = norm))
            self.conv.append(GCNConv(hidden_dim, num_class, norm = norm))
            # self.conv1 = GCNConv(feature_dim, hidden_dim, norm = 'both')
            # self.conv2 = GCNConv(hidden_dim, num_class, norm = 'both')
            self.relu = tlx.ReLU()
            self.dropout = tlx.layers.Dropout(drop_rate)

    def forward(self, x, edge_index, edge_weight, num_nodes):
        if self.num_layers == 1:
            x = self.conv[0](x, edge_index, edge_weight, num_nodes)
            return x
        for i in range(self.num_layers - 1):
            x = self.conv[i](x, edge_index, edge_weight, num_nodes)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.conv[-1](x, edge_index, edge_weight, num_nodes)
        # x = self.conv1(x, edge_index, edge_weight, num_nodes)
        # x = self.relu(x)
        # x = self.dropout(x)  # dropout not work in mindspore
        # x = self.conv2(x, edge_index, edge_weight, num_nodes)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 activation, k: int = 2, use_bn=False):
        super(GCN, self).__init__()
        assert k > 1, "k must > 1 !!"
        self.use_bn = use_bn
        self.k = k
        self.conv = tnn.ModuleList([tGCNConv(in_channels, hidden_channels)])
        self.bns = tnn.ModuleList([tnn.BatchNorm1d(hidden_channels)])
        for _ in range(1, k - 1):
            self.conv.append(tGCNConv(hidden_channels, hidden_channels))
            self.bns.append(tnn.BatchNorm1d(hidden_channels))
        self.conv.append(tGCNConv(hidden_channels, out_channels))
        # self.conv.append(tGCNConv(hidden_channels, hidden_channels))
        if activation is None:
            self.activation = F.relu
        else:
            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k - 1):
            # x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv[i](x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=0.5, training=self.training)
        return self.conv[-1](x, edge_index)


class GraphConvLayer(tnn.Module):
    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        super(GraphConvLayer, self).__init__()

        self.use_init = use_init
        self.use_weight = use_weight
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = tnn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index, x0):
        N = x.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm_in = (1. / d[col]).sqrt()
        d_norm_out = (1. / d[row]).sqrt()
        value = torch.ones_like(row) * d_norm_in * d_norm_out
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        x = matmul(adj, x)  # [N, D]

        if self.use_init:
            x = torch.cat([x, x0], 1)
            x = self.W(x)
        elif self.use_weight:
            x = self.W(x)

        return x


class GraphConv(tnn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, use_bn=True,
                 use_residual=True, use_weight=True, use_init=False, use_act=True):
        super(GraphConv, self).__init__()

        self.convs = tnn.ModuleList()
        self.fcs = tnn.ModuleList()
        self.fcs.append(tnn.Linear(in_channels, hidden_channels))

        self.bns = tnn.ModuleList()
        self.bns.append(tnn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(
                GraphConvLayer(hidden_channels, hidden_channels, use_weight, use_init))
            self.bns.append(tnn.BatchNorm1d(hidden_channels))
        self.classifier = tnn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0])
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
            # layer_.append(x)
        return self.classifier(x)
        # return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 activation, n_heads=8, k: int = 2, use_bn=False):
        super(GAT, self).__init__()
        assert k > 1, "k must > 1 !!"
        self.use_bn = use_bn
        self.k = k
        self.conv = tnn.ModuleList([GATConv(in_channels, hidden_channels// n_heads, heads=n_heads, dropout=0.6)])
        self.bns = tnn.ModuleList([tnn.BatchNorm1d(hidden_channels)])
        for _ in range(1, k - 1):
            self.conv.append(GATConv(hidden_channels, hidden_channels // n_heads, heads=n_heads, dropout=0.6))
            self.bns.append(tnn.BatchNorm1d(hidden_channels))
        self.conv.append(GATConv(hidden_channels, out_channels, heads=8, concat=False, dropout=0.6))
        # self.conv.append(tGCNConv(hidden_channels, hidden_channels))
        self.activation = F.relu

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k - 1):
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv[i](x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
        x = F.dropout(x, p=0.6, training=self.training)
        return self.conv[-1](x, edge_index)


