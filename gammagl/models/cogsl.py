import tensorlayerx as tlx
from gammagl.layers.conv import GCNConv
import tensorlayerx.nn as nn
import numpy as np
from scipy.sparse import coo_matrix
from scipy.special import softmax


class CoGSLModel(nn.Module):
    r"""CoGSL Model proposed in '"Compact Graph Structure Learning via Mutual Information Compression"
        <https://arxiv.org/pdf/2201.05540.pdf>'_ paper.
            
        Parameters
        ----------
        num_feature: int
            input feature dimension.
        cls_hid: int
            Classification hidden dimension.
        num_class: int
            number of classes.
        gen_hid: int
            GenView hidden dimension.
        mi_hid: int
            Mi_NCE hidden dimension.
        com_lambda_v1: float
            hyperparameter used to generate estimated view 1.
        com_lambda_v2: float
            hyperparameter used to generate estimated view 2.
        lam: float
            hyperparameter used to fusion views.
        alpha: float
            hyperparameter used to fusion views.
        cls_dropout: float
            Classification dropout rate.
        ve_dropout: float
            View_Estimator dropout rate.
        tau: float
            hyperparameter used to generate sim_matrix to get mi loss.
        ggl: bool
            whether to use gcnconv of gammagl.
        big: bool
            whether the dataset is too big.
        batch: int
            determine the sampling size when the dataset is too big.

    """

    def __init__(self, num_feature, cls_hid, num_class, gen_hid, mi_hid,
                 com_lambda_v1, com_lambda_v2, lam, alpha, cls_dropout, ve_dropout, tau, ggl, big, batch):
        super(CoGSLModel, self).__init__()
        self.cls = Classification(num_feature, cls_hid, num_class, cls_dropout, ggl)
        self.ve = View_Estimator(num_feature, gen_hid, com_lambda_v1, com_lambda_v2, ve_dropout, ggl)
        self.mi = MI_NCE(num_feature, mi_hid, tau, ggl, big, batch)
        self.fusion = Fusion(lam, alpha)

    def get_view(self, data):
        new_v1, new_v2 = self.ve(data)
        return new_v1, new_v2

    def get_mi_loss(self, feat, views):
        mi_loss = self.mi(views, feat)
        return mi_loss

    def get_cls_loss(self, v1, v2, feat):
        prob_v1 = self.cls(feat, v1, "v1")
        prob_v2 = self.cls(feat, v2, "v2")
        logits_v1 = tlx.log(prob_v1 + 1e-8)
        logits_v2 = tlx.log(prob_v2 + 1e-8)
        return logits_v1, logits_v2, prob_v1, prob_v2

    def get_v_cls_loss(self, v, feat):
        logits = tlx.log(self.cls(feat, v, "v") + 1e-8)
        return logits

    def get_fusion(self, v1, prob_v1, v2, prob_v2):
        v = self.fusion(v1, prob_v1, v2, prob_v2)
        return v


# base
class GCN_two(nn.Module):
    def __init__(self, input_dim, hid_dim1, hid_dim2, dropout=0., activation="relu"):
        super(GCN_two, self).__init__()
        self.conv1 = GCN_one(input_dim, hid_dim1)
        self.conv2 = GCN_one(hid_dim1, hid_dim2)

        self.dropout = tlx.layers.Dropout(dropout)
        assert activation in ["relu", "leaky_relu", "elu"]
        if activation == 'relu':
            self.activation = nn.ReLU()
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        if activation == 'elu':
            self.activation = nn.ELU()

    def forward(self, feature, adj):
        x1 = self.activation(self.conv1(feature, adj))
        x1 = self.dropout(x1)
        x2 = self.conv2(x1, adj)
        return x2


class GCN_one(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True, activation=None):
        super(GCN_one, self).__init__()
        self.fc = nn.Linear(in_features=in_ft, out_features=out_ft, W_init='xavier_uniform')
        self.activation = activation
        if bias:
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1, out_ft), init=initor)
        else:
            self.register_parameter('bias', None)


    def forward(self, feat, adj):
        feat = self.fc(feat)
        out = tlx.matmul(adj, feat)
        if self.bias is not None:
            out += self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out


class GCN_two_ggl(nn.Module):
    def __init__(self, input_dim, hid_dim1, hid_dim2, dropout=0., activation="relu"):
        super(GCN_two_ggl, self).__init__()
        self.conv1 = GCNConv(input_dim, hid_dim1)
        self.conv2 = GCNConv(hid_dim1, hid_dim2)

        self.dropout = tlx.layers.Dropout(dropout)
        assert activation in ["relu", "leaky_relu", "elu"]
        if activation == 'relu':
            self.activation = nn.ReLU()
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        if activation == 'elu':
            self.activation = nn.ELU()

    def forward(self, feature, adj):
        adj = tlx.convert_to_numpy(adj)
        non_zero_rows, non_zero_cols = np.nonzero(adj)
        edge_index = tlx.convert_to_tensor([non_zero_rows, non_zero_cols])
        x1 = self.activation(self.conv1(feature, edge_index))
        x1 = self.dropout(x1)
        x2 = self.conv2(x1, edge_index)
        return x2

class GCN_one_ggl(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True, activation=None):
        super(GCN_one_ggl, self).__init__()
        self.conv1 = GCNConv(in_ft, out_ft)
        self.activation = activation
        if bias:
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1, out_ft), init=initor)
        else:
            self.register_parameter('bias', None)

    def forward(self, feat, adj):
        adj = tlx.convert_to_numpy(adj)
        non_zero_rows, non_zero_cols = np.nonzero(adj)
        edge_index = tlx.convert_to_tensor([non_zero_rows, non_zero_cols])
        out = self.conv1(feat, edge_index)
        if self.bias is not None:
            out += self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out

# cls
class Classification(nn.Module):
    def __init__(self, num_feature, cls_hid, num_class, dropout, ggl):
        super(Classification, self).__init__()
        if ggl==False:
            self.encoder_v1 = GCN_two(num_feature, cls_hid, num_class, dropout)
            self.encoder_v2 = GCN_two(num_feature, cls_hid, num_class, dropout)
            self.encoder_v = GCN_two(num_feature, cls_hid, num_class, dropout)
        else:
            self.encoder_v1 = GCN_two_ggl(num_feature, cls_hid, num_class, dropout)
            self.encoder_v2 = GCN_two_ggl(num_feature, cls_hid, num_class, dropout)
            self.encoder_v = GCN_two_ggl(num_feature, cls_hid, num_class, dropout)

    def forward(self, feat, view, flag):
        if flag == "v1":
            prob = nn.Softmax()(self.encoder_v1(feat, view))
        elif flag == "v2":
            prob = nn.Softmax()(self.encoder_v2(feat, view))
        elif flag == "v":
            prob = nn.Softmax()(self.encoder_v(feat, view))
        return prob

# contrast
class Contrast:
    def __init__(self, tau):
        self.tau = tau

    def sim(self, z1, z2):
        z1_norm = tlx.sqrt(tlx.reduce_sum(tlx.square(z1), axis=1, keepdims=True))
        z2_norm = tlx.sqrt(tlx.reduce_sum(tlx.square(z2), axis=1, keepdims=True))
        dot_numerator = tlx.matmul(z1, tlx.transpose(z2))
        dot_denominator = tlx.matmul(z1_norm, tlx.transpose(z2_norm))
        sim_matrix = tlx.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix


    def cal(self, z1_proj, z2_proj):
        matrix_z1z2 = self.sim(z1_proj, z2_proj)
        matrix_z2z1 = tlx.transpose(matrix_z1z2)

        matrix_z1z2 = matrix_z1z2 / (tlx.reshape(tlx.reduce_sum(matrix_z1z2, axis=1), [-1,1]) + 1e-8)
        lori_v1v2 = -tlx.reduce_mean(tlx.log(tlx.diag(matrix_z1z2)+1e-8))

        matrix_z2z1 = matrix_z2z1 / (tlx.reshape(tlx.reduce_sum(matrix_z2z1, axis=1), [-1, 1]) + 1e-8)
        lori_v2v1 = -tlx.reduce_mean(tlx.log(tlx.diag(matrix_z2z1)+1e-8))
        return (lori_v1v2 + lori_v2v1) / 2

# fusion
class Fusion(nn.Module):
    def __init__(self, lam, alpha):
        super(Fusion, self).__init__()
        self.lam = lam
        self.alpha = alpha

    def get_weight(self, prob):
        out, _ = tlx.topk(prob, 2, dim=1, largest=True, sorted=True)
        fir = out[:, 0]
        sec = out[:, 1]
        w = tlx.exp(self.alpha*(self.lam*tlx.log(fir+1e-8) + (1-self.lam)*tlx.log(fir-sec+1e-8)))
        return w

    def forward(self, v1, prob_v1, v2, prob_v2):
        w_v1 = self.get_weight(prob_v1)
        w_v2 = self.get_weight(prob_v2)
        beta_v1 = w_v1 / (w_v1 + w_v2)
        beta_v2 = w_v2 / (w_v1 + w_v2)
        beta_v1 = tlx.reshape(beta_v1, (-1,1))
        beta_v2 = tlx.reshape(beta_v2, (-1,1))
        v = beta_v1 * v1 + beta_v2 * v2
        return v

# mi_nce
class MI_NCE(nn.Module):
    def __init__(self, num_feature, mi_hid, tau, ggl, big, batch):
        super(MI_NCE, self).__init__()
        if ggl == False:
            self.gcn = GCN_one(num_feature, mi_hid, activation=nn.PRelu())
            self.gcn1 = GCN_one(num_feature, mi_hid, activation=nn.PRelu())
            self.gcn2 = GCN_one(num_feature, mi_hid, activation=nn.PRelu())
        else:
            self.gcn = GCN_one_ggl(num_feature, mi_hid, activation=nn.PRelu())
            self.gcn1 = GCN_one_ggl(num_feature, mi_hid, activation=nn.PRelu())
            self.gcn2 = GCN_one_ggl(num_feature, mi_hid, activation=nn.PRelu())

        self.proj = nn.Sequential(
            nn.Linear(in_features=mi_hid, out_features=mi_hid),
            nn.ELU(),
            nn.Linear(in_features=mi_hid, out_features=mi_hid)
        )
        self.con = Contrast(tau)
        self.big = big
        self.batch = batch

    def forward(self, views, feat):
        v_emb = self.proj(self.gcn(feat, views[0]))
        v1_emb = self.proj(self.gcn1(feat, views[1]))
        v2_emb = self.proj(self.gcn2(feat, views[2]))
        # if dataset is so big, we will randomly sample part of nodes to perform MI estimation
        if self.big == True:
            idx = np.random.choice(feat.shape[0], self.batch, replace=False)
            idx.sort()
            v_emb = v_emb[idx]
            v1_emb = v1_emb[idx]
            v2_emb = v2_emb[idx]

        vv1 = self.con.cal(v_emb, v1_emb)
        vv2 = self.con.cal(v_emb, v2_emb)
        v1v2 = self.con.cal(v1_emb, v2_emb)

        return vv1, vv2, v1v2

# view_estimator
class GenView(nn.Module):
    def __init__(self, num_feature, hid, com_lambda, dropout, ggl):
        super(GenView, self).__init__()
        if ggl == False:
            self.gen_gcn = GCN_one(num_feature, hid, activation=nn.ReLU())
        else:
            self.gen_gcn = GCN_one_ggl(num_feature, hid, activation=nn.ReLU())
        self.gen_mlp = nn.Linear(in_features=2 * hid, out_features=1, W_init='xavier_normal')
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(axis=1)

        self.com_lambda = com_lambda
        self.dropout = tlx.layers.Dropout(dropout)

    def forward(self, v_ori, feat, v_indices, num_node):
        emb = self.gen_gcn(feat, v_ori)
        f1 = tlx.gather(emb, v_indices[0])
        f2 = tlx.gather(emb, v_indices[1])
        ff = tlx.concat([f1, f2], axis=1)
        temp = tlx.reshape(self.gen_mlp(self.dropout(ff)), (-1,))

        coo = coo_matrix( (tlx.convert_to_numpy(temp.cpu()), (v_indices[0].cpu(), v_indices[1].cpu())), shape= (num_node, num_node))
        dense = coo.todense()
        dense[dense == 0] = np.NINF
        pi = tlx.convert_to_tensor(softmax(dense,axis=1))

        gen_v = v_ori + self.com_lambda * pi
        return gen_v


class View_Estimator(nn.Module):
    def __init__(self, num_feature, gen_hid, com_lambda_v1, com_lambda_v2, dropout, ggl):
        super(View_Estimator, self).__init__()
        self.v1_gen = GenView(num_feature, gen_hid, com_lambda_v1, dropout, ggl)
        self.v2_gen = GenView(num_feature, gen_hid, com_lambda_v2, dropout, ggl)

    def forward(self, data):
        new_v1 = self.normalize(data['name'], self.v1_gen(data['view1'], data['x'], data['v1_indice'], data['num_nodes']))
        new_v2 = self.normalize(data['name'], self.v2_gen(data['view2'], data['x'], data['v2_indice'], data['num_nodes']))
        return new_v1, new_v2

    def normalize(self, dataset, adj):
        if dataset in ["wikics", "ms", "citeseer"]:
            adj_ = (adj + tlx.transpose(adj))
            normalized_adj = adj_
        else:
            adj_ = (adj + tlx.transpose(adj))
            normalized_adj = self._normalize(adj_ + tlx.convert_to_tensor(np.eye(adj_.shape[0]), dtype=tlx.float32))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = tlx.reduce_sum(mx, axis=1) + 1e-6  # avoid NaN
        r_inv = tlx.pow(rowsum, -1/2)
        r_inv = tlx.convert_to_tensor(r_inv)
        r_mat_inv = tlx.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx

