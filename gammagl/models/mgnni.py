import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.layers.conv import MGNNI_m_iter


class MGNNI_m_MLP(tlx.nn.Module):
    r"""
    Multiscale Graph Neural Networks with Implicit Layers proposed in `MGNNI: Multiscale Graph Neural Networks with Implicit Layers`_.

    .. _MGNNI: Multiscale Graph Neural Networks with Implicit Layers:
        https://arxiv.org/abs/2210.08353

    This model is for datasets chameleon and squirrel.

    Parameters
    ----------
        m (int): input feature dimension
        m_y (int): number of classes
        nhid (int): hidden dimension
        ks (list): list of scales
        threshold (int): threshold for convergence
        max_iter (str): maximum number of iterative solver iterations
        gamma (str): contraction Factor
        dropout (float): dropout rate (1 - keep probability)
        batch_norm (bool): whether to use batch norm
    """

    def __init__(self, m, m_y, nhid, ks, threshold, max_iter, gamma, dropout=0.5, batch_norm=False):
        super(MGNNI_m_MLP, self).__init__()
        self.fc1 = tlx.layers.Linear(out_features=nhid,
                                     in_features=m,
                                     # W_init='xavier_uniform',
                                     b_init=None)
        self.fc2 = tlx.layers.Linear(out_features=nhid,
                                     in_features=nhid,
                                     # W_init='xavier_uniform',
                                     )
        self.dropout = nn.Dropou(dropout)
        self.MGNNI_layer = MGNNI_m_att(nhid, m_y, ks, threshold, max_iter, gamma, dropout=self.dropout,
                                          batch_norm=batch_norm)

    def forward(self, X, edge_index, edge_weight=None, num_nodes=None):
        X = self.dropout(tlx.ops.transpose(X))
        X = nn.ReLU()(self.fc1(X))
        X = nn.Dropout(p=self.dropout)(X)
        X = self.fc2(X)
        output = self.MGNNI_layer(tlx.ops.transpose(X), edge_index, edge_weight, num_nodes)

        return output

class MGNNI_m_att(nn.Module):
    r"""
    Multiscale Graph Neural Networks with Implicit Layers proposed in
    `MGNNI: Multiscale Graph Neural Networks with Implicit Layers`_.

    .. _MGNNI: Multiscale Graph Neural Networks with Implicit Layers:
        https://arxiv.org/abs/2210.08353

    This model is for datasets cornell, texas and wisconsin.

    Parameters
    ----------
        m (int): input feature dimension
        m_y (int): number of classes
        ks (list): list of scales
        threshold (int): threshold for convergence
        max_iter (str): maximum number of iterative solver iterations
        gamma (str): contraction Factor
        dropout (float): dropout rate (1 - keep probability)
    """
    def __init__(self, m, m_y, ks, threshold, max_iter, gamma, dropout=0.5,
                 layer_norm=False, batch_norm=False):
        super(MGNNI_m_att, self).__init__()
        self.dropout = tlx.layers.Dropout(p=dropout)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.MGNNIs = nn.ModuleList()
        self.att = Attention(in_size=m)
        for k in ks:
            self.MGNNIs.append(MGNNI_m_iter(m, k, threshold, max_iter, gamma, layer_norm=layer_norm))
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(num_features=m, momentum=0.8)

        # self.B = nn.Parameter(1. / np.sqrt(m))
        tmp = tlx.convert_to_numpy(tlx.random_uniform((m_y, m)))
        # m = tlx.convert_to_tensor(m)
        tmp = tlx.convert_to_tensor(1. / np.sqrt(m) * tmp)

        self.B = nn.Parameter(tmp)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.MGNNIs)):
            self.MGNNIs[i].reset_parameters()

    def get_att_vals(self, x, edge_index, edge_weight, num_nodes):
        outputs = []
        for idx, model in enumerate(self.MGNNIs):
            tmp = model(x, edge_index, edge_weight, num_nodes)
            outputs.append(tlx.ops.transpose(tmp))
        outputs = tlx.stack(outputs, axis=1)
        att_vals = self.att(outputs)
        return att_vals

    def forward(self, X, edge_index, edge_weight, num_nodes):
        outputs = []
        for idx, model in enumerate(self.MGNNIs):
            tmp = model(X, edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
            outputs.append(tlx.ops.transpose(tmp))
        outputs = tlx.stack(outputs, axis=1)
        att_vals = self.att(outputs)
        outputs = tlx.reduce_sum(outputs * att_vals, 1)

        if self.batch_norm:
            outputs = self.bn1(outputs)
        outputs = self.dropout(outputs)
        outputs = outputs @ tlx.ops.transpose(self.B)
        return outputs


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(out_features=hidden_size, in_features=in_size),
            nn.Tanh(),
            nn.Linear(out_features=1, in_features=hidden_size)
        )

    def forward(self, z):
        w = self.project(z)
        beta = tlx.softmax(w, axis=1)
        return beta
