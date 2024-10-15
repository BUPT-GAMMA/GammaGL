from typing import Optional, Tuple, Any

import tensorlayerx as tlx
import random
import scipy.sparse as sp
import tensorlayerx.nn as nn
from sklearn.metrics import f1_score
from gammagl.models import GCNModel, GCNIIModel, GATModel

def get_model(args, dataset):
    if args.gnn.lower() == "gcn":
        return GCNModel(feature_dim=dataset.num_node_features,
                        hidden_dim=args.hidden_dim,
                        num_class=dataset.num_classes,
                        drop_rate=args.droprate,
                        num_layers=2,
                        norm='both',
                        name="GCN")
    elif args.gnn.lower() == "gat":
        return GATModel(feature_dim=dataset.num_node_features,
                        hidden_dim=args.hidden_dim,
                        num_class=dataset.num_classes,
                        heads=8,
                        drop_rate=args.droprate,
                        num_layers=2,
                        name="GAT")
    elif args.gnn.lower() == "gcnii":
        return GCNIIModel(feature_dim=dataset.num_node_features,
                          hidden_dim=64,
                          num_class=dataset.num_classes,
                          drop_rate=args.droprate,
                          num_layers=1,
                          alpha=0.1,
                          beta=0.3,
                          lambd=0.3,
                          variant=True,
                          name="GCNII")
    else: 
        raise ValueError(f"Model {args.gnn} is not supported")

def calculate_acc(logits, y, metrics):
    """
    Args:
        logits: node logits
        y: node labels
        metrics: tensorlayerx.metrics

    Returns:
        rst
    """

    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst

def dense_to_sparse(dense_adj):
    sparse_adj = sp.coo_matrix(dense_adj)

    row = tlx.ops.convert_to_tensor(sparse_adj.row, dtype=tlx.int64)
    col = tlx.ops.convert_to_tensor(sparse_adj.col, dtype=tlx.int64)

    edge_index = tlx.stack([row, col], axis=0)

    return edge_index

def F1score(output, lables):
    preds = output.max(1)[1].type_as(lables)
    f1 = f1_score(preds, lables, average='macro')
    return f1

class AssignmentMatricsMLP(nn.Module):
    def __init__(self, input_dim, num_clusters, activation='relu'):
        super(AssignmentMatricsMLP, self).__init__()
        self.mlp = nn.Linear(out_features=num_clusters, in_features=input_dim, act=None)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        assignment_matrics = self.mlp(x)
        assignment_matrics = tlx.softmax(assignment_matrics, axis=-1)
        if self.act is not None:
            assignment_matrics = self.act(assignment_matrics)
        return assignment_matrics


def edge_index_to_csr_matrix(edge_index, num_nodes):
    coo_matrix = sp.coo_matrix((tlx.ones([edge_index.shape[1]]), (edge_index[0], edge_index[1])),
                               shape=(num_nodes, num_nodes), dtype=float)
    csr_matrix = coo_matrix.tocsr()

    return csr_matrix


def dense_mincut_pool(
    x: Any,
    adj: Any,
    s: Any,
    mask: Optional[Any] = None,
    temp: float = 1.0,
) -> Tuple[Any, Any, Any, Any]:
    EPS = 1e-10

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = tlx.softmax(s / temp if temp != 1.0 else s, axis=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = tlx.matmul(s.transpose(1, 2), x)
    out_adj = tlx.matmul(tlx.matmul(s.transpose(1, 2), adj), s)

    # MinCut regularization.
    mincut_num = _rank3_trace(out_adj)
    d_flat = tlx.einsum('ijk->ij', adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(
        tlx.matmul(tlx.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / (mincut_den + EPS))
    mincut_loss = tlx.reduce_mean(mincut_loss)

    # Orthogonality regularization.
    ss = tlx.matmul(s.transpose(1, 2), s)
    i_s = tlx.eye(k).type_as(ss)
    ss_norm = ss / (tlx.sqrt(tlx.reduce_sum(ss ** 2, axis=(-1, -2), keepdims=True)) + EPS)
    i_s_norm = i_s / (tlx.sqrt(tlx.reduce_sum(i_s ** 2, axis=(-1, -2), keepdims=True)) + EPS)
    ortho_loss = ss_norm - i_s_norm
    ortho_loss = tlx.reduce_mean(tlx.sqrt(tlx.reduce_sum(ortho_loss ** 2, axis=(-1, -2))))

    EPS = 1e-15

    return out, out_adj, mincut_loss, ortho_loss

def _rank3_trace(x: Any) -> Any:
    return tlx.einsum('ijj->i', x)


def _rank3_diag(x: Any) -> Any:
    eye = tlx.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1))

    return out



class CITModule:
    def __init__(self, clusters, p=0.):
        self.p=p
        self.clusters = clusters

    def DSU(self, h_embedding, h_clu, assignment_matrics):
        index = random.sample(range(0,h_embedding.shape[0]), int(self.p * h_embedding.shape[0]))
        tensor_mask = tlx.ones((h_embedding.shape[0],1))
        tensor_mask[index]=0


        random_floats = tlx.random_uniform(minval=0, maxval=h_clu.shape[0], shape=(h_embedding.shape[0],), dtype=tlx.float32)
        tensor_selectclu = tlx.cast(tlx.floor(random_floats), dtype=tlx.int64)
        Select = tlx.argmax(assignment_matrics, axis=1)
        tensor_selectclu[tensor_selectclu == Select] = h_clu.shape[0] - 1

        a1 = tlx.expand_dims(h_embedding, axis=0)
        a1 = tlx.tile(a1, [h_clu.shape[0], 1, 1])
        b1 = tlx.expand_dims(h_clu, axis=1)
        c = a1 - b1
        d = tlx.pow(c, 2)

        s = tlx.transpose(assignment_matrics)
        s = tlx.expand_dims(s, axis=1)
        tensor_var_clu = tlx.matmul(s, d).squeeze()
        tensor_std_clu = tlx.sqrt(tensor_var_clu + 1e-10)

        tensor_mean_emb = tlx.reduce_mean(h_embedding, axis=1, keepdims=True)
        tensor_std_emb = tlx.sqrt(tlx.reduce_variance(h_embedding, axis=1, keepdims=True))

        sigma_mean = tlx.sqrt(tlx.reduce_variance(tlx.reduce_mean(h_clu, axis=1, keepdims=True), axis=0))
        sigma_std = tlx.sqrt(tlx.reduce_variance(tensor_std_clu, axis=0) + 1e-10)

        tensor_beta = tensor_std_clu[tensor_selectclu] + tlx.random_normal(tensor_std_emb.shape)*sigma_std
        tensor_gama = h_clu[tensor_selectclu] + tlx.random_normal(tensor_std_emb.shape)*sigma_mean

        h_new = tensor_mask * h_embedding + (1-tensor_mask) * (((h_embedding - h_clu[Select]) / (tensor_std_clu[Select] + 1e-10)) * tensor_beta + tensor_gama)

        return h_new

    def forward(self, h_embedding, mlp):
        assignment_matrics = mlp(h_embedding)
        h_pool = tlx.matmul(tlx.transpose(assignment_matrics), h_embedding)
        h_new = self.DSU(h_embedding, h_pool, assignment_matrics)
        return assignment_matrics, h_new  