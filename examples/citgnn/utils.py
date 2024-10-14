from typing import Optional, Tuple

import torch
from torch import Tensor
import tensorlayerx as tlx
import random
import scipy.sparse as sp
import torch.nn as nn
from sklearn.metrics import f1_score
from gammagl.models import GCNModel, GCNIIModel, GATModel, APPNPModel

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
                          beta=0.5,
                          lambd=0.5,
                          variant=True,
                          name="GCNII")
    else: 
        raise ValueError(f"Model {args.gnn} is not supported")
    

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def dense_to_sparse(dense_adj):
    sparse_adj = sp.coo_matrix(dense_adj)

    row = torch.tensor(sparse_adj.row, dtype=torch.long)
    col = torch.tensor(sparse_adj.col, dtype=torch.long)
    
    edge_index = torch.stack([row, col], dim=0)
    
    return edge_index

def F1score(output, lables):
    preds = output.max(1)[1].type_as(lables)
    f1 = f1_score(preds, lables, average='macro')
    return f1
    

def reassign_masks(graph, train_ratio=0.2, val_ratio=0.1, test_ratio=0.7):
    num_nodes = graph.num_nodes
    indices = list(range(num_nodes))
    random.shuffle(indices)

    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    test_size = num_nodes - train_size - val_size

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    return train_mask, val_mask, test_mask

class AssignmentMatricsMLP(nn.Module):
    def __init__(self, input_dim, num_clusters, activation='relu'):
        super(AssignmentMatricsMLP, self).__init__()
        self.mlp = nn.Linear(input_dim, num_clusters)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None
    
    def forward(self, x):
        assignment_matrics = self.mlp(x)
        assignment_matrics = torch.softmax(assignment_matrics, dim=-1)
        if self.act is not None:
            assignment_matrics = self.act(assignment_matrics)
        return assignment_matrics


def edge_index_to_csr_matrix(edge_index, num_nodes):
    coo_matrix = sp.coo_matrix((tlx.ones([edge_index.shape[1]]), (edge_index[0], edge_index[1])),
                               shape=(num_nodes, num_nodes), dtype=float)
    csr_matrix = coo_matrix.tocsr()

    return csr_matrix


def dense_mincut_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    temp: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    EPS = 1e-10
    
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # MinCut regularization.
    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum('ijk->ij', adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / (mincut_den + EPS))
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / (torch.norm(ss, dim=(-1, -2), keepdim=True) + EPS) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    EPS = 1e-15

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, mincut_loss, ortho_loss

def _rank3_trace(x: Tensor) -> Tensor:
    return torch.einsum('ijj->i', x)


def _rank3_diag(x: Tensor) -> Tensor:
    eye = torch.eye(x.size(1)).type_as(x)
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
        
        
        tensor_selectclu = torch.randint(low=0, high=h_clu.shape[0]-1, size=(h_embedding.shape[0],), dtype=torch.int64)
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