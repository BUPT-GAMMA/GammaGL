import tensorlayerx as tlx
from tensorlayerx import nn
from gammagl.layers.conv import GCNConv
from gammagl.utils import add_self_loops, calc_gcn_norm, degree, to_undirected, to_scipy_sparse_matrix
from gammagl.utils import to_dense_adj
import numpy as np
import scipy.sparse as sp
import pickle
from scipy.sparse import coo_matrix
import os.path as osp
from gammagl.mpops import gspmm


class Grace_POT_Encoder(tlx.nn.Module):
    def __init__(self, in_channels, out_channels, activation,
                 base_model = GCNConv, k = 2):
        super(Grace_POT_Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x, edge_index):
        for i in range(self.k):
            x = self.activation()(self.conv[i](x, edge_index))
        return x


class Grace_POT_Model(tlx.nn.Module):
    def __init__(self, encoder, num_hidden, num_proj_hidden,
                 tau = 0.5, dataset = "Cora", cached = "./"):
        super(Grace_POT_Model, self).__init__()
        self.encoder = encoder
        self.tau = tau
        self.dataset = dataset

        self.fc1 = tlx.nn.Linear(in_features=num_hidden, out_features=num_proj_hidden)
        self.fc2 = tlx.nn.Linear(in_features=num_proj_hidden, out_features=num_hidden)
        self.cached = cached

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

    def projection(self, z):
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

    def batched_semi_loss(self, z1, z2, batch_size):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        num_nodes = int(tlx.get_tensor_shape(z1)[0])
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: tlx.exp(x / self.tau)
        indices = tlx.arange(0, num_nodes)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-tlx.log(
                tlx.diag(between_sim[:, i * batch_size:(i + 1) * batch_size])
                / (tlx.reduce_sum(refl_sim, axis=1) + tlx.reduce_sum(between_sim, axis=1)
                   - tlx.diag(refl_sim[:, i * batch_size:(i + 1) * batch_size]))))

        return tlx.concat(losses)

    def loss(self, z1, z2, mean = True, batch_size = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)
        ret = (l1 + l2) * 0.5
        ret = tlx.reduce_mean(ret) if mean else tlx.reduce_sum(ret)

        return ret

    def pot_loss(self, z1, z2, x, edge_index, edge_index_1, local_changes=5, node_list = None, A_upper=None, A_lower=None):
        deg = tlx.to_device(degree(to_undirected(edge_index)[1]), "CPU")
        deg = tlx.convert_to_numpy(deg)
        A = to_scipy_sparse_matrix(edge_index).tocsr()
        A_tilde = A + sp.eye(A.shape[0])
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
            An = to_dense_adj(edge_index=new_edge_index, edge_attr=An)[0]
            An = tlx.convert_to_numpy(tlx.to_device(An, "CPU"))
            A_lower = np.zeros_like(An)
            A_lower[np.diag_indices_from(A_lower)] = np.diag(An)
            A_lower = np.float32(A_lower)
            self.upper_lower_file = osp.join(self.cached, f"{self.dataset}_{local_changes}_upper_lower.pkl")
            with open(self.upper_lower_file, 'wb') as file:
                pickle.dump((A_upper, A_lower), file)

        N = len(node_list)
        A_upper_mat = A_upper[node_list][:, node_list]
        A_lower_mat = A_lower[node_list][:, node_list]
        A_add = coo_matrix((A_upper_mat + A_lower_mat) / 2)
        A_sub = coo_matrix((A_upper_mat - A_lower_mat) / 2)
        edge_index_add = tlx.convert_to_tensor(np.array([A_add.row, A_add.col]), dtype=tlx.int64)
        weight_add = tlx.convert_to_tensor(A_add.data, dtype=tlx.float32)
        edge_index_sub = tlx.convert_to_tensor(np.array([A_sub.row, A_sub.col]), dtype=tlx.int64)
        weight_sub = tlx.convert_to_tensor(A_sub.data, dtype=tlx.float32)

        # get pre-activation bounds for each node
        XW = conv[0].linear(x)[node_list]
        H = self.encoder.activation()(conv[0](x, edge_index))
        HW = conv[1].linear(H)[node_list]
        W_1 = XW
        b1 = conv[0].bias
        z1_U = gspmm(edge_index_add, weight_add, W_1) + gspmm(edge_index_sub, weight_sub, tlx.abs(W_1)) + b1
        z1_L = gspmm(edge_index_add, weight_add, W_1) - gspmm(edge_index_sub, weight_sub, tlx.abs(W_1)) + b1
        W_2 = HW
        b2 = conv[1].bias
        z2_U = gspmm(edge_index_add, weight_add, W_2) + gspmm(edge_index_sub, weight_sub, tlx.abs(W_2)) + b2
        z2_L = gspmm(edge_index_add, weight_add, W_2) - gspmm(edge_index_sub, weight_sub, tlx.abs(W_2)) + b2


        # CROWN weights
        activation = self.encoder.activation
        alpha = 0 if activation == tlx.nn.ReLU else float(activation.trainable_weights[0])
        z2_norm = tlx.ops.l2_normalize(z2)
        z2_sum = tlx.reduce_sum(z2_norm, axis=0)
        Wcl = z2_norm * (N / (N-1)) - z2_sum / (N - 1)
        W_tilde_1, b_tilde_1, W_tilde_2, b_tilde_2 = get_crown_weights(z1_L, z1_U, z2_L, z2_U, alpha, gcn_weights, Wcl)

        # return the pot_score 
        num_nodes = A.shape[0]
        XW_tilde = tlx.reshape((tlx.matmul(x[node_list, None, :], W_tilde_1[:, :, None])), [-1, 1]) # N * 1
        edge_index_ptb_sl, _ = add_self_loops(edge_index_1, num_nodes=num_nodes)
        An_ptb = calc_gcn_norm(edge_index_ptb_sl, num_nodes=num_nodes)
        row, col = tlx.convert_to_numpy(edge_index_ptb_sl)
        An_ptb = coo_matrix((tlx.to_device(An_ptb, 'CPU'), (row, col)), shape=(num_nodes, num_nodes))
        An_ptb_csr = An_ptb.tocsr()
        selected_rows = An_ptb_csr[node_list, :]
        selected_rows_csc = selected_rows.tocsc()
        selected = selected_rows_csc[:, node_list].tocoo()
        row = selected.row
        col = selected.col
        data = selected.data
        edge_index = tlx.stack([tlx.convert_to_tensor(row, dtype=tlx.int64), tlx.convert_to_tensor(col, dtype=tlx.int64)], axis=0)
        H_tilde = gspmm(edge_index, tlx.convert_to_tensor(data), XW_tilde) + tlx.reshape(b_tilde_1, [-1, 1])
        pot_score = gspmm(edge_index, tlx.convert_to_tensor(data), H_tilde) + tlx.reshape(b_tilde_2, [-1, 1])
        pot_score = tlx.squeeze(pot_score, axis=1)
        target = tlx.zeros(tlx.get_tensor_shape(pot_score)) + 1
        pot_loss = tlx.losses.sigmoid_cross_entropy(pot_score, target)
        return pot_loss
    
    @property
    def cache(self):
        return self.upper_lower_file
    
def get_alpha_beta(l, u, alpha):
    alpha_L = tlx.zeros(tlx.get_tensor_shape(l))
    alpha_U = tlx.zeros(tlx.get_tensor_shape(l))
    beta_L = tlx.zeros(tlx.get_tensor_shape(l))
    beta_U = tlx.zeros(tlx.get_tensor_shape(l))
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
    W_tilde_2 = tlx.matmul(Lambda_2, tlx.transpose(W2_tensor))
    b_tilde_2 = tlx.diag(tlx.matmul(Lambda_2, tlx.transpose(Delta_2 + b2_tensor)))
    lambda_1 = tlx.where(W_tilde_2 >= 0, alpha_1_L, alpha_1_U)
    Delta_1 = tlx.where(W_tilde_2 >= 0, beta_1_L, beta_1_U)
    Lambda_1 = lambda_1 * W_tilde_2
    W_tilde_1 = tlx.matmul(Lambda_1, tlx.transpose(W1_tensor))
    b_tilde_1 = tlx.diag(tlx.matmul(Lambda_1, tlx.transpose(Delta_1 + b1_tensor)))
    return W_tilde_1, b_tilde_1, W_tilde_2, b_tilde_2

