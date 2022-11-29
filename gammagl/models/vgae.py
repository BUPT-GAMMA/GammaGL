# 开发时间: 2022/11/1 17:49

import tensorlayerx as tlx
import tensorlayerx.nn as nn

from gammagl.layers.conv import GCNConv


class VGAEModel(tlx.nn.Module):
    def __init__(self, feature_dim, hidden1_dim, hidden2_dim, drop_rate=0., num_layers=2, norm='none', name=None):
        super().__init__(name=name)
        #self.num_layers = num_layers
        #self.conv = nn.ModuleList([GCNConv(feature_dim, hidden1_dim, norm=norm)])
        #self.conv.append(GCNConv(hidden1_dim, hidden2_dim, norm=norm))
        self.conv1 = GCNConv(feature_dim, hidden1_dim, norm=norm)
        self.conv_mu = GCNConv(hidden1_dim, hidden2_dim, norm=norm)
        self.conv_logstd = GCNConv(hidden1_dim, hidden2_dim, norm=norm)
        self.dc = InnerProductDecoder()

        self.relu = tlx.ReLU()
        self.dropout = tlx.layers.Dropout(drop_rate)

    def encode(self, x, edge_index, edge_weight, num_nodes):
        x = self.conv1(x, edge_index, edge_weight, num_nodes)
        x = self.relu(x)
        x = self.dropout(x)
        mu = self.conv_mu(x, edge_index, edge_weight, num_nodes)
        logstd = self.conv_logstd(x, edge_index, edge_weight, num_nodes)

        return mu, logstd

    def reparameterize(self, mu, logstd):
        std = tlx.exp(logstd)
        eps = tlx.random_normal((std.shape[0], std.shape[1]))
        z = mu + tlx.multiply(std, eps)        #对应位置元素相乘
        return z

    def forward(self, x, edge_index, edge_weight, num_nodes):
        mu, logstd = self.encode(x, edge_index, edge_weight, num_nodes)
        z = self.reparameterize(mu, logstd)
        return self.dc(z), mu, logstd

#可修改
'''
    def forward(self, x, edge_index, edge_weight, num_nodes):

        x = self.conv1(x, edge_index, edge_weight, num_nodes)
        x = self.relu(x)
        x = self.dropout(x)
        mu = self.conv_mu(x, edge_index, edge_weight, num_nodes)
        logstd = self.conv_logstd(x, edge_index, edge_weight, num_nodes)

        #x = self.conv[0](x, edge_index, edge_weight, num_nodes)
        #x = self.relu(x)
        #x = self.dropout(x)
        #mu = self.conv[1](x, edge_index, edge_weight, num_nodes)
        #logstd = self.conv[1](x, edge_index, edge_weight, num_nodes)

        return mu, logstd
'''


class GAEModel(tlx.nn.Module):
    def __init__(self, feature_dim, hidden1_dim, hidden2_dim, drop_rate=0., num_layers=2, norm='none', name=None):
        super().__init__(name=name)
        #self.num_layers = num_layers
        #self.conv = nn.ModuleList([GCNConv(feature_dim, hidden1_dim, norm=norm)])
        #self.conv.append(GCNConv(hidden1_dim, hidden2_dim, norm=norm))
        self.conv1 = GCNConv(feature_dim, hidden1_dim, norm=norm)
        self.conv2 = GCNConv(hidden1_dim, hidden2_dim, norm=norm)
        self.dc = InnerProductDecoder()

        self.relu = tlx.ReLU()
        self.dropout = tlx.layers.Dropout(drop_rate)

    def encode(self, x, edge_index, edge_weight, num_nodes):
        x = self.conv1(x, edge_index, edge_weight, num_nodes)
        x = self.relu(x)
        x = self.dropout(x)
        mu = self.conv2(x, edge_index, edge_weight, num_nodes)

        return mu

    def forward(self, x, edge_index, edge_weight, num_nodes):
        mu = self.encode(x, edge_index, edge_weight, num_nodes)
        return self.dc(mu), mu, 1


class InnerProductDecoder(tlx.nn.Module):

    def __init__(self, drop_rate=0.):
        super(InnerProductDecoder, self).__init__()
        self.sigmoid = tlx.Sigmoid()
        self.dropout = tlx.layers.Dropout(drop_rate)

    def forward(self, z):
        z = self.dropout(z)
        zt = tlx.transpose(z)
        adj = tlx.matmul(z, zt)
        #adj = self.sigmoid(adj)
        return adj
