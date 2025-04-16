from tqdm import tqdm
import tensorlayerx as tlx
import math


class MultiHeadAttention(tlx.nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, channels, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.channels = channels
        d_q = d_k = d_v = channels // n_head

        self.w_qs = tlx.nn.Linear(in_features=channels, out_features=channels, b_init=None)
        self.w_ks = tlx.nn.Linear(in_features=channels, out_features=channels, b_init=None)
        self.w_vs = tlx.nn.Linear(in_features=channels, out_features=channels, b_init=None)
        self.fc = tlx.nn.Linear(in_features=channels, out_features=channels, b_init=None)
        
        self.temperature = d_k ** 0.5
        self.dropout = tlx.nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        n_head = self.n_head
        d_q = d_k = d_v = self.channels // n_head
        
        # Get batch size and sequence length
        q_shape = tlx.get_tensor_shape(q)
        k_shape = tlx.get_tensor_shape(k)
        v_shape = tlx.get_tensor_shape(v)
        
        B_q, N_q = q_shape[0], q_shape[1]
        B_k, N_k = k_shape[0], k_shape[1]
        B_v, N_v = v_shape[0], v_shape[1]

        residual = q

        # Linear projections and reshape
        q = tlx.reshape(self.w_qs(q), shape=[B_q, N_q, n_head, d_q])
        k = tlx.reshape(self.w_ks(k), shape=[B_k, N_k, n_head, d_k])
        v = tlx.reshape(self.w_vs(v), shape=[B_v, N_v, n_head, d_v])

        # Transpose for attention
        q = tlx.transpose(q, perm=[0, 2, 1, 3])
        k = tlx.transpose(k, perm=[0, 2, 1, 3])
        v = tlx.transpose(v, perm=[0, 2, 1, 3])

        # Scaled Dot-Product Attention
        attn = tlx.matmul(q / self.temperature, tlx.transpose(k, perm=[0, 1, 3, 2]))

        if mask is not None:
            mask = tlx.expand_dims(mask, axis=1)
            attn = tlx.where(mask == 0, tlx.ones_like(attn) * (-1e9), attn)

        attn = self.dropout(tlx.softmax(attn, axis=-1))
        output = tlx.matmul(attn, v)

        # Transpose back and reshape
        output = tlx.transpose(output, perm=[0, 2, 1, 3])
        output = tlx.reshape(output, shape=[B_q, N_q, -1])
        
        # Final linear projection
        output = self.fc(output)
        output = output + residual

        return output, attn


class FFN(tlx.nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, channels, dropout=0.1):
        super(FFN, self).__init__()
        self.lin1 = tlx.nn.Linear(in_features=channels, out_features=channels)  # position-wise
        self.lin2 = tlx.nn.Linear(in_features=channels, out_features=channels)  # position-wise
        self.layer_norm = tlx.nn.LayerNorm(normalized_shape=channels, epsilon=1e-6)
        self.dropout = tlx.nn.Dropout(p=dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = tlx.relu(self.lin1(x))
        x = self.lin2(x) + residual

        return x


class BGALayer(tlx.nn.Module):
    def __init__(self, n_head, channels, use_patch_attn=True, dropout=0.1):
        super(BGALayer, self).__init__()
        self.node_norm = tlx.nn.LayerNorm(normalized_shape=channels, epsilon=1e-6)
        self.node_transformer = MultiHeadAttention(n_head, channels, dropout)
        self.patch_norm = tlx.nn.LayerNorm(normalized_shape=channels, epsilon=1e-6)
        self.patch_transformer = MultiHeadAttention(n_head, channels, dropout)
        self.node_ffn = FFN(channels, dropout)
        self.patch_ffn = FFN(channels, dropout)
        self.fuse_lin = tlx.nn.Linear(in_features=2 * channels, out_features=channels)
        self.use_patch_attn = use_patch_attn

    def forward(self, x, patch, attn_mask=None, need_attn=False):
        x = self.node_norm(x)

        patch_x = x[patch]
        patch_x, attn = self.node_transformer(patch_x, patch_x, patch_x, attn_mask)
        patch_x = self.node_ffn(patch_x)

        if self.use_patch_attn:
            p = self.patch_norm(tlx.reduce_mean(patch_x, axis=1, keepdims=False))
            p = tlx.expand_dims(p, axis=0)
            p, _ = self.patch_transformer(p, p, p)
            p = self.patch_ffn(p)
            p = tlx.transpose(p, perm=[1, 0, 2])
            
            # repeat操作
            patch_shape = tlx.get_tensor_shape(patch)
            p = tlx.tile(p, [1, patch_shape[1], 1])
            
            z = tlx.concat([patch_x, p], axis=2)
            patch_x = tlx.relu(self.fuse_lin(z)) + patch_x

        x[patch] = patch_x

        return x