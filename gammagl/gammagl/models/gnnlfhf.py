from typing import List
import tensorlayerx as tlx
import tensorlayerx.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.sparse import coo_matrix
from gammagl.utils.norm import calc_gcn_norm
from gammagl.utils.num_nodes import maybe_num_nodes


def calc_LF_exact(edge_index, x, alpha, mu):
    nnodes = x.shape[0]
    edge_weight = calc_gcn_norm(edge_index, maybe_num_nodes(edge_index))
    edge_weight = tlx.convert_to_numpy(edge_weight)
    edge_index = tlx.convert_to_numpy(edge_index)
    sparse_matrix = coo_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(nnodes, nnodes))
    dense_matrix = tlx.convert_to_tensor(sparse_matrix.toarray())
    A_inner = (1+ alpha * mu - alpha) * tlx.eye(nnodes) + (2 * alpha - alpha * mu -1) * dense_matrix
    A_outer = mu * tlx.eye(nnodes) + (1 - mu) * dense_matrix
    A_inner = tlx.convert_to_numpy(A_inner)
    A_inner = np.linalg.inv(A_inner)
    A_inner = tlx.convert_to_tensor(A_inner)
    return alpha * tlx.matmul(A_inner, A_outer)

def calc_HF_exact(edge_index, x, alpha, beta):
    nnodes = x.shape[0]
    edge_weight = calc_gcn_norm(edge_index, maybe_num_nodes(edge_index))
    edge_weight = tlx.convert_to_numpy(edge_weight)
    edge_index = tlx.convert_to_numpy(edge_index)
    sparse_matrix = coo_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(nnodes, nnodes))
    dense_matrix = tlx.convert_to_tensor(sparse_matrix.toarray())
    L = tlx.eye(nnodes) - dense_matrix
    A_inner = alpha * tlx.eye(nnodes) + (alpha * beta + 1  - alpha)  * L
    A_outer = tlx.eye(nnodes) + beta * L
    A_inner = tlx.convert_to_numpy(A_inner)
    A_inner = np.linalg.inv(A_inner)
    A_inner = tlx.convert_to_tensor(A_inner)
    return alpha * tlx.matmul(A_inner, A_outer)

class LFExact(nn.Module):
    def __init__(self, edge_index, x, alpha, mu, drop_prob = None):
        super().__init__()

        LF_mat = calc_LF_exact(edge_index, x, alpha, mu)
        self.register_buffer('mat', tlx.convert_to_tensor(LF_mat))

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, predictions):
        return tlx.matmul(self.dropout(self.mat), predictions)

class HFExact(nn.Module):
    def __init__(self, edge_index, x, alpha, beta, drop_prob = None):
        super().__init__()

        HF_mat = calc_HF_exact(edge_index, x, alpha, beta)
        self.register_buffer('mat', tlx.convert_to_tensor(HF_mat))

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, predictions):
        return tlx.matmul(self.dropout(self.mat), predictions)

class LFPowerIteration(nn.Module):
    def __init__(self, edge_index, x, alpha, mu, niter, drop_prob = None):
        super().__init__()
        self.alpha = alpha
        self.niter = niter
        self.mu = mu
        edge_weight = calc_gcn_norm(edge_index, maybe_num_nodes(edge_index))
        nnodes = x.shape[0]
        edge_weight = tlx.convert_to_numpy(edge_weight)
        edge_index = tlx.convert_to_numpy(edge_index)
        sparse_matrix = coo_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(nnodes, nnodes))
        dense_matrix = sparse_matrix.toarray()
        self.register_buffer('A_hat', tlx.convert_to_tensor((1/(1 + alpha * mu - alpha)) * dense_matrix))

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, local_preds):
        preds = (self.mu / (1 + self.alpha * self.mu - self.alpha)) * local_preds + (1 - self.mu) * tlx.matmul(self.A_hat, local_preds)
        local_preds = self.alpha * preds
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = (1 - 2 * self.alpha + self.mu * self.alpha) * tlx.matmul(A_drop, preds) + local_preds
        return preds

class HFPowerIteration(nn.Module):
    def __init__(self, edge_index, x, alpha, beta, niter, drop_prob = None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.niter = niter

        edge_weight = calc_gcn_norm(edge_index, maybe_num_nodes(edge_index))
        nnodes = x.shape[0]
        sparse_matrix = coo_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(nnodes, nnodes))
        dense_matrix = sparse_matrix.toarray()
        L = sp.eye(nnodes).toarray() - dense_matrix
        self.register_buffer('L_hat', tlx.convert_to_tensor(L, dtype=tlx.float32))
        self.register_buffer('A_hat', tlx.convert_to_tensor(((alpha * beta + 1 - alpha)/(alpha * beta + 1)) * dense_matrix))

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, local_preds):
        preds = 1/(self.alpha * self.beta + 1) * local_preds + (self.beta/(self.alpha * self.beta + 1)) * tlx.matmul(self.L_hat, local_preds)
        local_preds = self.alpha * preds
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = tlx.matmul(A_drop, preds) + local_preds
        return preds


class GNNLFHFModel(nn.Module):
    r"""GNN with Low-pass Filtering Kernel and High-pass Filtering
        Kernel proposed in `"Interpreting and Unifying Graph Neural
        Networks with An Optimization Framework"
        <https://arxiv.org/pdf/2101.11859>`_ paper.
        
        Parameters
        ----------
        in_channels: int
            input feature dimension.
        out_channels: int
            output feature dimension.
        hidden_dim: int
            hidden dimension.
        model_type: str
            the model type.
        model_form: str
            the model form.
        edge_index: tensor
            the structure of graph.
        x: tensor
            the feature matrix of graph.
        alpha: float
            the value of alpha.
        mu: float
            the value of mu.
        beta: float
            the value of beta.
        niter: int
            the value of niter.
        drop_rate: float, optional
            dropout rate.
        num_layers: int, optional
            number of layers.
        name: str, optional
            model name.

    """

    def __init__(self, in_channels, out_channels, hidden_dim, 
                 model_type, model_form, edge_index, x, alpha, 
                 mu, beta, niter, drop_rate = 0.2, num_layers = 2,
                 name = None):
        super().__init__(name=name)

        fcs = [nn.Linear(in_features=in_channels, out_features=hidden_dim)]
        for i in range(2, num_layers):
            fcs.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        fcs.append(nn.Linear(in_features=hidden_dim, out_features=out_channels))
        self.fcs = nn.ModuleList(fcs)

        self.reg_params = list(self.fcs[0].parameters())

        self.dropout = nn.Dropout(drop_rate)
        self.relu = nn.ReLU()

        if model_type == "GNN-LF":
            if model_form == "closed":
                self.propagation = LFExact(edge_index, x, alpha=alpha, mu=mu, drop_prob=drop_rate)
            elif model_form == "iterative":
                self.propagation = LFPowerIteration(edge_index, x, alpha=alpha, mu=mu, niter=niter, drop_prob=drop_rate)
        elif model_type == "GNN-HF":
            if model_form == "closed":
                self.propagation = HFExact(edge_index, x, alpha=alpha, beta=beta, drop_prob=drop_rate)
            elif model_form == "iterative":
                self.propagation = HFPowerIteration(edge_index, x, alpha=alpha, beta=beta, niter=niter, drop_prob=drop_rate)

    def forward(self, x):
        layer_inner = self.relu(self.fcs[0](self.dropout(x)))
        for fc in self.fcs[1:-1]:
            layer_inner = self.relu(fc(layer_inner))
        local_logits = self.fcs[-1](self.dropout(layer_inner))
        final_logits = self.propagation(local_logits)
        return tlx.logsoftmax(final_logits, dim=-1)

