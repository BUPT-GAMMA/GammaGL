from tensorlayerx import nn
from gammagl.utils import degree
import tensorlayerx as tlx
import numpy as np
from .edge_encoder import EdgeEncoding


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in, dim_q, dim_k, edge_dim, max_path_distance):
        super().__init__()

        self.edge_encoding = EdgeEncoding(edge_dim, max_path_distance)

        self.q = nn.Linear(in_features=dim_in, out_features=dim_q)
        self.k = nn.Linear(in_features=dim_in, out_features=dim_k)
        self.v = nn.Linear(in_features=dim_in, out_features=dim_k)

    def forward(self, query, key, value, edge_attr, b, edge_paths, ptr):
        matrix = np.full((query.shape[0], query.shape[0]), fill_value=-1e6)
        batch_mask_neg_inf = tlx.convert_to_tensor(matrix)
        batch_mask_zeros = tlx.zeros(shape=(query.shape[0], query.shape[0]))

        if ptr == None:
            batch_mask_neg_inf = tlx.ones(shape=(query.shape[0], query.shape[0]))
            batch_mask_zeros += 1
        else:
            for i in range(len(ptr) - 1):
                batch_mask_neg_inf[ptr[i] : ptr[i + 1], ptr[i] : ptr[i + 1]] = 1
                batch_mask_zeros[ptr[i] : ptr[i + 1], ptr[i] : ptr[i + 1]] = 1

        query = self.q(query)
        key = self.k(key)
        value = self.v(value)
        value = tlx.convert_to_tensor(value, dtype=tlx.float32)

        c = self.edge_encoding(query, edge_attr, edge_paths)
        ndims = len(tlx.get_tensor_shape(key))
        perm = [1, 0] + list(range(2, ndims))
        a = tlx.matmul(query, (tlx.transpose(key, perm))) / tlx.get_tensor_shape(query)[-1] ** 0.5
        a = (a + b + c) * batch_mask_neg_inf
        softmax = tlx.softmax(logits=a, axis=-1) * batch_mask_zeros
        x = tlx.matmul(softmax, value)
        return x


class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_in, dim_q, dim_k, edge_dim, max_path_distance):
        super().__init__()

        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k, edge_dim, max_path_distance) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(in_features=num_heads*dim_k, out_features=dim_in)

    def forward(self, x, edge_attr, b, edge_paths, ptr):
        x = tlx.concat([attention_head(x, x, x, edge_attr, b, edge_paths, ptr) for attention_head in self.heads], axis=-1)
        x = tlx.convert_to_tensor(x, dtype=tlx.float32)
        return self.linear(x)


class GraphormerLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, n_heads, max_path_distance):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads

        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=n_heads,
            edge_dim=edge_dim,
            max_path_distance=max_path_distance,
        )

        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)
        self.ff = nn.Linear(in_features=node_dim, out_features=node_dim)

    def forward(self, x, edge_attr, b, edge_paths, ptr):
        x_prime = self.attention(self.ln_1(x), edge_attr, b, edge_paths, ptr) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime

        return x_new
