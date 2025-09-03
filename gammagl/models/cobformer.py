import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.layers.conv import GCNConv
from gammagl.utils import mask_to_index

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = tlx.matmul(q / self.temperature, tlx.transpose(k, (0, 1, 3, 2)))

        if mask is not None:
            # Convert mask to appropriate type and apply
            # Use -1e9 for masked positions to ensure they get near-zero attention weights
            mask = tlx.cast(mask, dtype=attn.dtype)
            attn = tlx.where(mask == 0, tlx.ones_like(attn) * -1e9, attn)
        
        attn = self.dropout(tlx.softmax(attn, axis=-1))
        output = tlx.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, channels, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.channels = channels
        d_q = d_k = d_v = channels // n_head

        self.w_qs = nn.Linear(in_features=channels, out_features=channels, b_init=None)
        self.w_ks = nn.Linear(in_features=channels, out_features=channels, b_init=None)
        self.w_vs = nn.Linear(in_features=channels, out_features=channels, b_init=None)
        self.fc = nn.Linear(in_features=channels, out_features=channels, b_init=None)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        n_head = self.n_head
        d_q = d_k = d_v = self.channels // n_head
        B_q = tlx.get_tensor_shape(q)[0]
        N_q = tlx.get_tensor_shape(q)[1]
        B_k = tlx.get_tensor_shape(k)[0]
        N_k = tlx.get_tensor_shape(k)[1]
        B_v = tlx.get_tensor_shape(v)[0]
        N_v = tlx.get_tensor_shape(v)[1]

        residual = q

        # Pass through the pre-attention projection: B * N x (h*dv)
        # Separate different heads: B * N x h x dv
        q = tlx.reshape(self.w_qs(q), (B_q, N_q, n_head, d_q))
        k = tlx.reshape(self.w_ks(k), (B_k, N_k, n_head, d_k))
        v = tlx.reshape(self.w_vs(v), (B_v, N_v, n_head, d_v))

        # Transpose for attention dot product: B * h x N x dv
        q, k, v = tlx.transpose(q, (0, 2, 1, 3)), tlx.transpose(k, (0, 2, 1, 3)), tlx.transpose(v, (0, 2, 1, 3))

        # For head axis broadcasting.
        if mask is not None:
            mask = tlx.expand_dims(mask, 1)

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: B x N x h x dv
        # Combine the last two dimensions to concatenate all the heads together: B x N x (h*dv)
        q = tlx.reshape(tlx.transpose(q, (0, 2, 1, 3)), (B_q, N_q, -1))
        q = self.fc(q)
        q = q + residual

        return q, attn


class FFN(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, channels, dropout=0.1):
        super(FFN, self).__init__()
        self.lin1 = nn.Linear(in_features=channels, out_features=channels)
        self.lin2 = nn.Linear(in_features=channels, out_features=channels)
        self.layer_norm = nn.LayerNorm(normalized_shape=channels, epsilon=1e-6)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.Dropout(x)
        x = tlx.relu(self.lin1(x))
        x = self.lin2(x) + residual

        return x


class SimpleFFN(nn.Module):
    ''' A simple feed-forward module without LayerNorm and residual connection '''

    def __init__(self, in_channels, hidden_channels):
        super(SimpleFFN, self).__init__()
        self.lin1 = nn.Linear(in_features=in_channels, out_features=hidden_channels)
        self.lin2 = nn.Linear(in_features=hidden_channels, out_features=hidden_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(x)
        x = tlx.relu(self.lin1(x))
        # Return the output of the first linear layer as in the original implementation
        return x


class BGALayer(nn.Module):
    def __init__(self, n_head, channels, num_nodes, use_patch_attn=True, dropout=0.1):
        super(BGALayer, self).__init__()
        self.num_nodes = num_nodes
        self.node_norm = nn.LayerNorm(normalized_shape=channels)
        self.node_transformer = MultiHeadAttention(n_head, channels, dropout)
        self.patch_norm = nn.LayerNorm(normalized_shape=channels)
        self.patch_transformer = MultiHeadAttention(n_head, channels, dropout)
        self.node_ffn = FFN(channels, dropout)
        self.patch_ffn = FFN(channels, dropout)
        self.fuse_lin = nn.Linear(in_features=2 * channels, out_features=channels)
        self.use_patch_attn = use_patch_attn
        self.attn = None

    def forward(self, x, patch, attn_mask=None, need_attn=False):
        x = self.node_norm(x)
        
        # More efficient patch processing using gather and scatter operations
        patch_shape = tlx.get_tensor_shape(patch)
        
        # Flatten patch indices for efficient gathering
        patch_flat = tlx.reshape(patch, [-1])
        
        # Gather all patch features at once using the correct gather interface
        patch_features_flat = tlx.gather(x, patch_flat)
        
        # Reshape back to [num_patches, max_patch_size, feature_dim]
        patch_x = tlx.reshape(patch_features_flat, [patch_shape[0], patch_shape[1], -1])
        
        patch_x, attn = self.node_transformer(patch_x, patch_x, patch_x, attn_mask)
        patch_x = self.node_ffn(patch_x)
        
        if need_attn:
            # Note: This is a simplified version without the complex attention tracking
            self.attn = attn

        if self.use_patch_attn:
            # Mean pooling across patch dimension
            patch_mean = tlx.reduce_mean(patch_x, axis=1, keepdims=False)
            p = tlx.expand_dims(self.patch_norm(patch_mean), axis=0)
            p, _ = self.patch_transformer(p, p, p)
            p = tlx.transpose(self.patch_ffn(p), (1, 0, 2))
            
            # Repeat to match patch dimensions
            p = tlx.tile(p, (1, tlx.get_tensor_shape(patch_x)[1], 1))
            z = tlx.concat([patch_x, p], axis=2)
            patch_x = tlx.relu(self.fuse_lin(z)) + patch_x

        # More efficient scatter back to original tensor
        # Flatten patch_x for scattering
        patch_x_flat = tlx.reshape(patch_x, [-1, tlx.get_tensor_shape(patch_x)[-1]])
        
        # Flatten patch indices
        patch_flat = tlx.reshape(patch, [-1])
        
        # Update x at the patch node positions using tensor_scatter_nd_update
        x = tlx.tensor_scatter_nd_update(x, tlx.expand_dims(patch_flat, axis=-1), patch_x_flat)

        return x


class BGA(nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 layers: int, n_head: int, use_patch_attn=True, dropout1=0.5, dropout2=0.1):
        super(BGA, self).__init__()
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout1)
        self.attribute_encoder = SimpleFFN(in_channels, hidden_channels)
        self.BGALayers = nn.ModuleList()
        for _ in range(0, layers):
            self.BGALayers.append(
                BGALayer(n_head, hidden_channels, num_nodes, use_patch_attn, dropout=dropout2))
        self.classifier = nn.Linear(in_features=hidden_channels, out_features=out_channels)

    def forward(self, x, patch, need_attn=False):
        # Create attention mask more efficiently
        # Use boolean mask first, then convert to float for matmul
        patch_mask = patch != self.num_nodes - 1  # Boolean mask
        patch_mask = tlx.cast(patch_mask, dtype=tlx.float32)  # Convert to float
        patch_mask = tlx.expand_dims(patch_mask, axis=-1)  # Add dimension for matmul
        attn_mask = tlx.matmul(patch_mask, tlx.transpose(patch_mask, (0, 2, 1)))  # Create attention mask
        # Convert to int mask to match original implementation
        attn_mask = tlx.cast(attn_mask, dtype=tlx.int32)

        x = self.attribute_encoder(x)
        for i in range(0, self.layers):
            x = self.BGALayers[i](x, patch, attn_mask, need_attn)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 activation=None, k: int = 2, use_bn=False):
        super(GCN, self).__init__()
        assert k > 1, "k must > 1 !!"
        self.use_bn = use_bn
        self.k = k
        self.conv = nn.ModuleList([GCNConv(in_channels, hidden_channels)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features=hidden_channels)])
        for _ in range(1, k - 1):
            self.conv.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(num_features=hidden_channels))
        self.conv.append(GCNConv(hidden_channels, out_channels))
        self.dropout = nn.Dropout(p=0.5)

        if activation is None:
            self.activation = tlx.relu
        else:
            self.activation = activation

    def forward(self, x, edge_index):
        for i in range(self.k - 1):
            x = self.conv[i](x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        return self.conv[-1](x, edge_index)


class CoBFormerModel(nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 activation=None, gcn_layers: int = 2, gcn_type: int = 1, layers: int = 1, 
                 n_head: int = 4, dropout1=0.5, dropout2=0.1,
                 alpha=0.8, tau=0.5, gcn_use_bn=False, use_patch_attn=True, name=None):
        super(CoBFormerModel, self).__init__(name=name)
        self.alpha = alpha
        self.tau = tau
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.activation = activation
        self.dropout = nn.Dropout(dropout1)
        
        if gcn_type == 1:
            self.gcn = GCN(in_channels, hidden_channels, out_channels, activation, k=gcn_layers, use_bn=gcn_use_bn)
        else:
            # For simplicity, we'll use the same GCN for both types
            self.gcn = GCN(in_channels, hidden_channels, out_channels, activation, k=gcn_layers, use_bn=gcn_use_bn)
            
        self.bga = BGA(num_nodes, in_channels, hidden_channels, out_channels, layers, n_head,
                       use_patch_attn, dropout1, dropout2)

    def forward(self, x, patch, edge_index, need_attn=False):
        z1 = self.gcn(x, edge_index)
        z2 = self.bga(x, patch, need_attn)
        return z1, z2

    def loss(self, pred1, pred2, label, mask):
        # Convert one-hot label to indices if needed
        if len(tlx.get_tensor_shape(label)) > 1 and tlx.get_tensor_shape(label)[-1] > 1:
            label = tlx.argmax(label, axis=-1)
        
        # Convert mask to the same dtype as predictions to avoid type conversion issues
        mask = tlx.cast(mask, dtype=tlx.float32)
        
        # For labeled nodes - use proper tensor handling with masking
        l1 = tlx.losses.softmax_cross_entropy_with_logits(pred1, label)
        l1_masked = l1 * mask
        l1 = tlx.reduce_sum(l1_masked) / (tlx.reduce_sum(mask) + 1e-8)
        
        l2 = tlx.losses.softmax_cross_entropy_with_logits(pred2, label)
        l2_masked = l2 * mask
        l2 = tlx.reduce_sum(l2_masked) / (tlx.reduce_sum(mask) + 1e-8)
        
        pred1_scaled = pred1 * self.tau
        pred2_scaled = pred2 * self.tau
        
        # For unlabeled nodes, use softmax of the other prediction
        pred2_softmax = tlx.softmax(pred2_scaled, axis=-1)
        pred1_softmax = tlx.softmax(pred1_scaled, axis=-1)
        
        l3 = tlx.losses.softmax_cross_entropy_with_logits(pred1_scaled, pred2_softmax)
        not_mask = 1.0 - mask
        l3_masked = l3 * not_mask
        l3 = tlx.reduce_sum(l3_masked) / (tlx.reduce_sum(not_mask) + 1e-8)
        
        l4 = tlx.losses.softmax_cross_entropy_with_logits(pred2_scaled, pred1_softmax)
        l4_masked = l4 * not_mask
        l4 = tlx.reduce_sum(l4_masked) / (tlx.reduce_sum(not_mask) + 1e-8)
        
        loss = self.alpha * (l1 + l2) + (1 - self.alpha) * (l3 + l4)
        return loss