import os
os.environ.setdefault('TL_BACKEND', 'torch')

import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn
import torch
from gammagl.layers.conv import GCNConv
from gammagl.utils.negative_sampling import negative_sampling
from gammagl.mpops import unsorted_segment_mean


def dropout_edge(edge_index, p, training=True):
    """Randomly drop edges during training."""
    if p == 0.0 or not training:
        return edge_index, tlx.ones((edge_index.shape[1],), dtype=tlx.bool)
    mask_np = np.random.binomial(1, 1 - p, size=edge_index.shape[1]).astype(bool)
    indices = np.where(mask_np)[0]
    indices_tensor = tlx.convert_to_tensor(indices, dtype=tlx.int64)
    kept = tlx.gather(edge_index, indices_tensor, axis=1)
    mask = tlx.convert_to_tensor(mask_np, dtype=tlx.bool)
    return kept, mask


def align_feature_rows(feat, data, target_rows):
    """Align feature rows to the current mini-batch node count."""
    if feat.shape[0] == target_rows:
        return feat
    if hasattr(data, 'n_id') and data.n_id is not None:
        n_id = data.n_id
        if isinstance(n_id, torch.Tensor):
            n_id = n_id.to(dtype=torch.int64)
        else:
            if hasattr(n_id, 'detach'):
                n_id = n_id.detach().cpu().numpy()
            elif hasattr(n_id, 'numpy'):
                n_id = n_id.numpy()
            n_id = tlx.convert_to_tensor(n_id, dtype=tlx.int64)
        return tlx.gather(feat, n_id, axis=0)
    return feat[:target_rows]


class GCN(nn.Module):
    def __init__(self, n_layers, in_features, hidden_features, out_features, drop_edge=0.5, drop_feats=0.5):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(GCNConv(in_features, hidden_features))
        for _ in range(n_layers - 2):
            self.layers.append(GCNConv(hidden_features, hidden_features))
        self.layers.append(GCNConv(hidden_features, out_features))
        self.drop_edge = drop_edge
        self.drop = tlx.nn.Dropout(p=drop_feats)

    def forward(self, x, edge_index, num_nodes):
        edge = dropout_edge(edge_index, self.drop_edge, training=self.is_train)[0]
        for layer in self.layers[:-1]:
            x = self.drop(tlx.relu(layer(x, edge, num_nodes=num_nodes)))
        x = self.layers[-1](x, edge, num_nodes=num_nodes)
        return x


class NodeClsHead(nn.Module):
    def __init__(self, pretrained_model, in_dim, hidden_dim, num_cls, drop_edge, drop_feats):
        super(NodeClsHead, self).__init__()
        self.pretrained_model = pretrained_model
        self.head = GCN(2, in_dim, hidden_dim, num_cls, drop_edge=drop_edge, drop_feats=drop_feats)

    def forward(self, data):
        x_E, x_H, x_S, q_E, q_H, q_S, commit_loss_E, commit_loss_H, commit_loss_S = self.pretrained_model(data)
        
        manifold_H = self.pretrained_model.manifold_H
        manifold_S = self.pretrained_model.manifold_S
        manifold_E = self.pretrained_model.manifold_E
        x_e = manifold_E.logmap0(q_E)          
        x_h = manifold_H.logmap0(q_H)
        x_s = manifold_S.logmap0(q_S)
        dx = data.x.to(dtype=torch.float32) if isinstance(data.x, torch.Tensor) else tlx.convert_to_tensor(data.x, dtype=tlx.float32)
        target_rows = dx.shape[0]
        x_h = align_feature_rows(x_h, data, target_rows)
        x_s = align_feature_rows(x_s, data, target_rows)
        x_e = align_feature_rows(x_e, data, target_rows)
        x = tlx.concat([dx, x_h, x_s, x_e], axis=-1)
        num_nodes = x.shape[0]
        edge_index = data.edge_index.to(dtype=torch.int64) if isinstance(data.edge_index, torch.Tensor) else tlx.convert_to_tensor(data.edge_index, dtype=tlx.int64)
        return self.head(x, edge_index, num_nodes)


class LinkPredHead(nn.Module):
    def __init__(self, pretrained_model, in_dim, out_dim):
        super(LinkPredHead, self).__init__()
        self.pretrained_model = pretrained_model
        self.head = tlx.layers.Linear(in_features=in_dim, out_features=out_dim, b_init=None)

    def forward(self, data):
        x_E, x_H, x_S, q_E, q_H, q_S, _, _, _ = self.pretrained_model(data)
        manifold_H = self.pretrained_model.manifold_H
        manifold_S = self.pretrained_model.manifold_S    
        x_h = manifold_H.logmap0(q_H)
        x_s = manifold_S.logmap0(q_S)
        dx = data.x.to(dtype=torch.float32) if isinstance(data.x, torch.Tensor) else tlx.convert_to_tensor(data.x, dtype=tlx.float32)
        target_rows = dx.shape[0]
        x_h = align_feature_rows(x_h, data, target_rows)
        x_s = align_feature_rows(x_s, data, target_rows)
        q_E = align_feature_rows(q_E, data, target_rows)
        x = tlx.concat([dx, x_h, x_s, q_E], axis=-1)
        x = self.head(x)

        edge_label_index = data.edge_label_index.to(dtype=torch.int64) if isinstance(data.edge_label_index, torch.Tensor) else tlx.convert_to_tensor(data.edge_label_index, dtype=tlx.int64)
        x_src = x[edge_label_index[0]]
        x_dst = x[edge_label_index[1]]
        x_src_norm = tlx.sqrt(tlx.reduce_sum(x_src * x_src, axis=-1, keepdims=True))
        x_dst_norm = tlx.sqrt(tlx.reduce_sum(x_dst * x_dst, axis=-1, keepdims=True))
        score = tlx.reduce_sum(x_src * x_dst, axis=-1) / (x_src_norm.squeeze(-1) * x_dst_norm.squeeze(-1) + 1e-8)

        return score, data.edge_label


class GraphClsHead(nn.Module):
    def __init__(self, pretrained_model, in_dim, hidden_dim, num_cls, drop_edge, drop_feats):
        super(GraphClsHead, self).__init__()
        self.pretrained_model = pretrained_model
        self.head = tlx.layers.Linear(in_features=in_dim, out_features=num_cls, b_init=None)
        self.drop = tlx.nn.Dropout(p=drop_feats)

    def forward(self, data):
        x_E, x_H, x_S, q_E, q_H, q_S, _, _, _ = self.pretrained_model(data)
        manifold_H = self.pretrained_model.manifold_H
        manifold_S = self.pretrained_model.manifold_S
        x_h = manifold_H.logmap0(x_H)
        x_s = manifold_S.logmap0(x_S)
        dx = data.x.to(dtype=torch.float32) if isinstance(data.x, torch.Tensor) else tlx.convert_to_tensor(data.x, dtype=tlx.float32)
        target_rows = dx.shape[0]
        x_h = align_feature_rows(x_h, data, target_rows)
        x_s = align_feature_rows(x_s, data, target_rows)
        q_E = align_feature_rows(q_E, data, target_rows)
        x = tlx.concat([dx, x_h, x_s, q_E], axis=-1)
        batch = data.batch.to(dtype=torch.int64) if isinstance(data.batch, torch.Tensor) else tlx.convert_to_tensor(data.batch, dtype=tlx.int64)
        max_batch = int(tlx.convert_to_numpy(tlx.reduce_max(batch)))
        x = unsorted_segment_mean(x, batch, num_segments=max_batch + 1)
        return self.head(self.drop(x))


class ShotNCHead(nn.Module):
    def __init__(self, pretrained_model, cls_embeddings, in_dim, hidden_dim, cls_dim, drop_edge, drop_feats):
        super(ShotNCHead, self).__init__()
        self.pretrained_model = pretrained_model
        self.head = GCN(2, in_dim, hidden_dim, cls_dim, drop_edge=drop_edge, drop_feats=drop_feats)
        self.cls_embeddings = cls_embeddings

    def forward(self, data):
        x_E, x_H, x_S, q_E, q_H, q_S, _, _, _ = self.pretrained_model(data)
        manifold_H = self.pretrained_model.manifold_H
        manifold_S = self.pretrained_model.manifold_S
        manifold_E = self.pretrained_model.manifold_E
        x_e = manifold_E.logmap0(q_E)   
        x_h = manifold_H.logmap0(q_H)
        x_s = manifold_S.logmap0(q_S)
        dx = data.x.to(dtype=torch.float32) if isinstance(data.x, torch.Tensor) else tlx.convert_to_tensor(data.x, dtype=tlx.float32)
        target_rows = dx.shape[0]
        x_h = align_feature_rows(x_h, data, target_rows)
        x_s = align_feature_rows(x_s, data, target_rows)
        x_e = align_feature_rows(x_e, data, target_rows)
        x = tlx.concat([dx, x_h, x_s, x_e], axis=-1)
        edge_index = data.edge_index.to(dtype=torch.int64) if isinstance(data.edge_index, torch.Tensor) else tlx.convert_to_tensor(data.edge_index, dtype=tlx.int64)
        num_nodes = x.shape[0]
        x = self.head(x, edge_index, num_nodes)
        x_norm = tlx.sqrt(tlx.reduce_sum(x * x, axis=-1, keepdims=True))
        cls_norm = tlx.sqrt(tlx.reduce_sum(self.cls_embeddings * self.cls_embeddings, axis=-1, keepdims=True))
        out = tlx.matmul(x / (x_norm + 1e-8), tlx.transpose(self.cls_embeddings) / (cls_norm.T + 1e-8))
        return out
