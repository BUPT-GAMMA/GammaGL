from typing import Optional
import torch
import tensorlayerx as tlx
from tensorlayerx import nn
from torch.sparse import mm
from gammagl.layers.conv import GCNConv
from gammagl.utils import add_self_loops, calc_gcn_norm, degree, to_undirected, to_scipy_sparse_matrix
import numpy as np
import scipy.sparse as sp
import pickle
from scipy.sparse import coo_matrix
import os.path as osp
from time import perf_counter as t
from gammagl.mpops import unsorted_segment_sum
import tensorflow

def BCEWithLogitsLoss(output, target):
    return torch.nn.BCEWithLogitsLoss()(output, target)

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(
            in_features=ft_in, 
            out_features=nb_classes, 
            W_init=tlx.initializers.xavier_uniform(nb_classes)
        )

    def forward(self, seq):
        ret = self.fc(seq)
        ret = nn.LogSoftmax(dim=-1)(ret) 
        return ret

class Encoder(tlx.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x:tlx.convert_to_tensor, edge_index:tlx.convert_to_tensor):
        for i in range(self.k):
            x = self.activation()(self.conv[i](x, edge_index))
        return x


class Model(tlx.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5, dataset: str = "Cora"):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau
        self.dataset = dataset

        self.fc1 = tlx.nn.Linear(in_features=num_hidden, out_features=num_proj_hidden)
        self.fc2 = tlx.nn.Linear(in_features=num_proj_hidden, out_features=num_hidden)
    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

    def projection(self, z: tlx.convert_to_tensor)->tlx.convert_to_tensor:
        z = tlx.nn.activation.ELU()(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = tlx.l2_normalize(z1, axis=1)
        z2 = tlx.l2_normalize(z2, axis=1)
        return tlx.matmul(z1, tlx.transpose(z2))
    
    def semi_loss(self, z1, z2):
        f = lambda x: tlx.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        x1 = tlx.reduce_sum(refl_sim, axis=1) + tlx.reduce_sum(between_sim, axis=1) - tlx.diag(refl_sim, 0)
        loss = -tlx.log(tlx.diag(between_sim) / x1)

        return loss

    def batched_semi_loss(self, z1, z2,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: tlx.exp(x / self.tau)
        indices = tlx.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-tlx.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return tlx.concat(losses)

    def loss(self, z1, z2,
             mean: bool = True, batch_size: int = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)
        ret = (l1 + l2) * 0.5
        ret = tlx.reduce_mean(ret) if mean else np.sum(ret)

        return ret
    def pot_loss(self, z1, z2, x, edge_index, edge_index_1, local_changes=5, node_list = None, A_upper=None, A_lower=None):
        deg = degree(to_undirected(edge_index)[1]).cpu().numpy()
        A = to_scipy_sparse_matrix(edge_index).tocsr()
        A_tilde = A + sp.eye(A.shape[0])
        assert self.encoder.k == 2 # only support 2-layer GCN
        conv = self.encoder.conv
        # W1, b1 = conv[0].all_weights, conv[0].bias
        W1, b1 = tlx.transpose(conv[0].linear.weights), conv[0].bias
        W2, b2 = tlx.transpose(conv[1].linear.weights), conv[1].bias
        gcn_weights = [W1, b1, W2, b2]
        # load entry-wise bounds, if not exist, calculate
        if A_upper is None:
            degs_tilde = deg + 1
            max_delete = np.maximum(degs_tilde.astype("int") - 2, 0)
            max_delete = np.minimum(max_delete, np.round(local_changes * deg)) # here
            sqrt_degs_tilde_max_delete = 1 / np.sqrt(degs_tilde - max_delete)
            A_upper = sqrt_degs_tilde_max_delete * sqrt_degs_tilde_max_delete[:, None]
            A_upper = np.where(A_tilde.toarray() > 0, A_upper, np.zeros_like(A_upper))
            A_upper = np.float32(A_upper)
            #new_edge_index, An = calc_gcn_norm(edge_index, num_nodes=A.shape[0])
            new_edge_index ,_ =add_self_loops(edge_index, num_nodes=A.shape[0])
            An = calc_gcn_norm(new_edge_index, num_nodes=A.shape[0])
            An = to_dense_adj(edge_index=new_edge_index, edge_attr=An)[0].cpu().numpy()
            A_lower = np.zeros_like(An)
            A_lower[np.diag_indices_from(A_lower)] = np.diag(An)
            A_lower = np.float32(A_lower)
            # upper_lower_file = osp.join(osp.expanduser('~/datasets'),f"bounds/{self.dataset}_{local_changes}_upper_lower.pkl")
            # upper_lower_file = open(f'datasets/bounds/{self.dataset}_{local_changes}_upper_lower.pkl', 'wb'))
            upper_lower_file = osp.join(osp.expanduser('~/datasets'), f"bounds/{self.dataset}_{local_changes}_upper_lower.pkl")

            if self.dataset == 'ogbn-arxiv':
                with open(upper_lower_file, 'wb') as file:
                    pickle.dump((tlx.convert_to_tensor(A_upper).to_sparse(), tlx.convert_to_tensor(A_lower).to_sparse()), file)
            else:
                with open(upper_lower_file, 'wb') as file:
                    pickle.dump((A_upper, A_lower), file)
        N = len(node_list)
        if self.dataset == 'ogbn-arxiv':
            A_upper_tensor = tlx.convert_to_tensor(A_upper.to_dense()[node_list][:,node_list]).to_sparse()
            A_lower_tensor = tlx.convert_to_tensor(A_lower.to_dense()[node_list][:,node_list]).to_sparse()
        else:
            A_upper_tensor = tlx.convert_to_tensor(A_upper[node_list][:,node_list]).to_sparse()
            A_lower_tensor = tlx.convert_to_tensor(A_lower[node_list][:,node_list]).to_sparse()
        # get pre-activation bounds for each node
        XW = conv[0].linear(x)[node_list]
        H = self.encoder.activation()(conv[0](x, edge_index))
        HW = conv[1].linear(H)[node_list]
        W_1 = XW
        b1 = conv[0].bias
        z1_U = mm((A_upper_tensor + A_lower_tensor) / 2, W_1) + mm((A_upper_tensor - A_lower_tensor) / 2, tlx.abs(W_1)) + b1
        z1_L = mm((A_upper_tensor + A_lower_tensor) / 2, W_1) - mm((A_upper_tensor - A_lower_tensor) / 2, tlx.abs(W_1)) + b1
        W_2 = HW
        b2 = conv[1].bias
        z2_U = mm((A_upper_tensor + A_lower_tensor) / 2, W_2) + mm((A_upper_tensor - A_lower_tensor) / 2, tlx.abs(W_2)) + b2
        z2_L = mm((A_upper_tensor + A_lower_tensor) / 2, W_2) - mm((A_upper_tensor - A_lower_tensor) / 2, tlx.abs(W_2)) + b2
        # CROWN weights
        activation = self.encoder.activation
        alpha = 0 if activation == tlx.nn.ReLU else activation.weight.item()
        z2_norm = tlx.ops.l2_normalize(z2)
        z2_sum = z2_norm.sum(axis=0)
        Wcl = z2_norm * (N / (N-1)) - z2_sum / (N - 1)
        W_tilde_1, b_tilde_1, W_tilde_2, b_tilde_2 = get_crown_weights(z1_L, z1_U, z2_L, z2_U, alpha, gcn_weights, Wcl)
        # return the pot_score 
        XW_tilde = (x[node_list,None,:] @ W_tilde_1[:,:,None]).reshape(-1,1) # N * 1
        edge_index_ptb_sl ,_ =add_self_loops(edge_index_1, num_nodes=A.shape[0])
        An_ptb=calc_gcn_norm(edge_index_ptb_sl, num_nodes=A.shape[0])
        row, col = tlx.convert_to_numpy(edge_index_ptb_sl)
        An_ptb = coo_matrix((An_ptb.cpu(),(row, col)), shape=(A.shape[0],A.shape[0])).toarray()
        An_ptb=tlx.gather(tlx.convert_to_tensor(An_ptb),tlx.convert_to_tensor(node_list),0)
        An_ptb=tlx.gather(An_ptb,tlx.convert_to_tensor(node_list),1)
        An_ptb=tlx.convert_to_tensor(An_ptb)
        H_tilde = mm(An_ptb, XW_tilde) + b_tilde_1.reshape(-1,1)
        pot_score = mm(An_ptb, H_tilde) + b_tilde_2.reshape(-1,1)
        pot_score = tlx.squeeze(pot_score,axis=1)
        target = tlx.zeros(tlx.get_tensor_shape(pot_score)) + 1
        pot_loss = BCEWithLogitsLoss(pot_score, target)
        return pot_loss
    
def get_alpha_beta(l, u, alpha):
    alpha_L= tlx.zeros(l.shape)
    alpha_U= tlx.zeros(l.shape)
    beta_L= tlx.zeros(l.shape)
    beta_U= tlx.zeros(l.shape)
    pos_mask = l >= 0
    neg_mask = u <= 0
    alpha_L[pos_mask] = 1
    alpha_U[pos_mask] = 1
    alpha_L[neg_mask] = alpha
    alpha_U[neg_mask] = alpha
    not_mask = ~(pos_mask | neg_mask)
    alpha_not_upp = u[not_mask] - alpha * l[not_mask]
    alpha_not = alpha_not_upp / (u[not_mask] - l[not_mask])
    alpha_L[not_mask] = alpha_not
    alpha_U[not_mask] = alpha_not
    beta_U[not_mask] = (alpha - 1) * u[not_mask] * l[not_mask] / alpha_not_upp
    return alpha_L, alpha_U, beta_L, beta_U

def get_crown_weights(l1, u1, l2, u2, alpha, gcn_weights, Wcl):
    alpha_2_L, alpha_2_U, beta_2_L, beta_2_U = get_alpha_beta(l2, u2, alpha) # onehop
    alpha_1_L, alpha_1_U, beta_1_L, beta_1_U = get_alpha_beta(l1, u1, alpha) # twohop
    lambda_2 = tlx.where(Wcl >= 0, alpha_2_L, alpha_2_U) # N * d
    Delta_2 = tlx.where(Wcl >= 0, beta_2_L, beta_2_U) # N * d
    Lambda_2 = lambda_2 * Wcl # N * d
    W1_tensor, b1_tensor, W2_tensor, b2_tensor = gcn_weights
    W_tilde_2 = Lambda_2 @ W2_tensor
    b_tilde_2 = tlx.diag(Lambda_2 @ tlx.transpose(Delta_2 + b2_tensor))
    lambda_1 = tlx.where(W_tilde_2 >= 0, alpha_1_L, alpha_1_U)
    Delta_1 = tlx.where(W_tilde_2 >= 0, beta_1_L, beta_1_U)
    Lambda_1 = lambda_1 * W_tilde_2
    W_tilde_1 = Lambda_1 @ W1_tensor
    b_tilde_1 = tlx.diag(Lambda_1 @ tlx.transpose(Delta_1 + b1_tensor))
    return W_tilde_1, b_tilde_1, W_tilde_2, b_tilde_2

def to_dense_adj(
    edge_index:tlx.convert_to_tensor,
    batch:tlx.convert_to_tensor = None,
    edge_attr:tlx.convert_to_tensor = None,
    max_num_nodes: Optional[int] = None,
):

    if batch is None:
        num_nodes = int(tlx.reduce_max(edge_index)) + 1 if tlx.numel(edge_index) > 0 else 0
        batch = tlx.zeros([num_nodes], dtype=tlx.int64)

    batch_size = int(tlx.reduce_max(batch)) + 1 if tlx.numel(batch) > 0 else 1
    one = tlx.ones(shape=(batch.size(0),), dtype=tlx.int64)
    num_nodes = unsorted_segment_sum(one, batch, batch_size)
    cum_nodes = tlx.concat([tlx.zeros([1],dtype=tlx.int64), tlx.cumsum(num_nodes)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = num_nodes.max().item()
        # max_num_nodes = int(tlx.reduce_max(num_nodes))
    elif ((tlx.numel(idx1) > 0 and tlx.reduce_max(idx1) >= max_num_nodes)
          or (tlx.numel(idx2) > 0 and tlx.reduce_max(idx2) >= max_num_nodes)):
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = tlx.ones(tlx.numel(idx0), dtype=tlx.int64)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.size())[1:]
    adj = tlx.zeros(size, dtype=tlx.int64)

    flattened_size = batch_size * max_num_nodes * max_num_nodes
    adj = adj.reshape([flattened_size] + list(adj.size())[3:])
    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    adj = unsorted_segment_sum(edge_attr, idx)
    adj = adj.reshape(size)

    return adj