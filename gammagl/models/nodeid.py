import tensorlayerx as tlx
import tensorlayerx.nn as nn
import numpy as np

from gammagl.layers.conv import GATConv, GCNConv


def l2norm(tensor):
    return tlx.l2_normalize(tensor, axis=-1)


def one_hot(indices, depth):
    return tlx.ops.OneHot(depth=depth)(indices)


class VectorQuantize(tlx.nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        commitment_weight=0.25,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=2,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code

        init_weight = tlx.initializers.XavierUniform()
        self.embed = tlx.cast(init_weight((codebook_size, dim)), tlx.float32)
        self.embed_avg = np.array(tlx.convert_to_numpy(self.embed), copy=True)
        self.cluster_size = np.zeros((codebook_size,), dtype=np.float32)

    def _ema_update_codebook(self, flat_x, embed_onehot):
        flat_x_np = tlx.convert_to_numpy(flat_x).astype(np.float32)
        onehot_np = tlx.convert_to_numpy(embed_onehot).astype(np.float32)

        counts = onehot_np.sum(axis=0)
        embed_sum = onehot_np.T @ flat_x_np

        self.cluster_size = self.cluster_size * self.decay + (1.0 - self.decay) * counts
        self.embed_avg = self.embed_avg * self.decay + (1.0 - self.decay) * embed_sum

        cluster_size_sum = float(self.cluster_size.sum())
        if cluster_size_sum > 0:
            cluster_size = ((self.cluster_size + self.eps) /
                            (cluster_size_sum + self.codebook_size * self.eps)) * cluster_size_sum
        else:
            cluster_size = np.ones_like(self.cluster_size, dtype=np.float32)

        new_embed = self.embed_avg / np.expand_dims(np.maximum(cluster_size, self.eps), axis=-1)

        dead_mask = self.cluster_size < self.threshold_ema_dead_code
        dead_count = int(dead_mask.sum())
        if dead_count > 0:
            replace = flat_x_np.shape[0] < dead_count
            sample_idx = np.random.choice(flat_x_np.shape[0], dead_count, replace=replace)
            new_embed[dead_mask] = flat_x_np[sample_idx]
            self.embed_avg[dead_mask] = new_embed[dead_mask]
            self.cluster_size[dead_mask] = self.threshold_ema_dead_code

        self.embed = tlx.convert_to_tensor(new_embed.astype(np.float32), dtype=tlx.float32)

    def forward(self, x):
        only_one = len(tlx.get_tensor_shape(x)) == 2
        if only_one:
            x = tlx.expand_dims(x, axis=1)

        shape = tlx.get_tensor_shape(x)
        flat_x = tlx.reshape(x, (-1, self.dim))

        flat_x_norm = l2norm(flat_x)
        embed_norm = l2norm(self.embed)
        sim = tlx.matmul(flat_x_norm, tlx.transpose(embed_norm))

        embed_ind = tlx.argmax(sim, axis=-1)
        embed_onehot = tlx.cast(one_hot(embed_ind, self.codebook_size), tlx.float32)
        quantize = tlx.matmul(embed_onehot, self.embed)
        quantize = tlx.reshape(quantize, shape)

        if self.is_train:
            self._ema_update_codebook(flat_x, embed_onehot)

        if self.is_train:
            quantize = x + tlx.detach(quantize - x)

        commit_loss = tlx.reduce_mean(tlx.square(tlx.detach(quantize) - x))
        loss = commit_loss * self.commitment_weight

        embed_ind = tlx.reshape(embed_ind, (shape[0], shape[1]))

        if only_one:
            quantize = tlx.squeeze(quantize, axis=1)
            embed_ind = tlx.squeeze(embed_ind, axis=1)

        return quantize, embed_ind, loss


class ResidualVectorQuant(tlx.nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_res_layers=3,
        commitment_weight=0.25,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=2,
    ):
        super().__init__()
        self.vq_layers = nn.ModuleList(
            [
                VectorQuantize(
                    dim=dim,
                    codebook_size=codebook_size,
                    commitment_weight=commitment_weight,
                    decay=decay,
                    eps=eps,
                    threshold_ema_dead_code=threshold_ema_dead_code,
                )
                for _ in range(num_res_layers)
            ]
        )

    def forward(self, x):
        quantized_outputs = []
        total_loss = 0
        embed_indices = []

        residual = x
        for vq_layer in self.vq_layers:
            quantized, embed_ind, layer_loss = vq_layer(residual)
            total_loss = total_loss + layer_loss
            embed_indices.append(embed_ind)
            quantized_outputs.append(quantized)
            residual = residual - quantized

        output = quantized_outputs[0]
        for i in range(1, len(quantized_outputs)):
            output = output + quantized_outputs[i]

        return output, embed_indices, total_loss


class NodeIDGNN(tlx.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        local_layers=3,
        in_dropout=0.0,
        dropout=0.5,
        heads=1,
        pre_ln=False,
        kmeans=1,
        num_codes=16,
        gnn='gat',
        vq_decay=0.8,
        vq_eps=1e-5,
        vq_dead_code_threshold=2,
    ):
        super().__init__()
        self.in_drop = nn.Dropout(in_dropout)
        self.dropout = nn.Dropout(dropout)
        self.pre_ln = pre_ln
        self.kmeans = kmeans
        self.hidden_dim = hidden_channels * heads
        self.gnn = gnn

        self.vqs = nn.ModuleList()
        self.local_convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = nn.ModuleList()

        for _ in range(local_layers):
            if gnn == 'gat':
                self.local_convs.append(
                    GATConv(
                        in_channels=self.hidden_dim,
                        out_channels=hidden_channels,
                        heads=heads,
                        concat=True,
                        dropout_rate=dropout,
                        add_bias=False,
                    )
                )
            else:
                self.local_convs.append(
                    GCNConv(
                        in_channels=self.hidden_dim,
                        out_channels=self.hidden_dim,
                        norm='both',
                    )
                )

            self.vqs.append(
                ResidualVectorQuant(
                    dim=self.hidden_dim,
                    codebook_size=num_codes,
                    num_res_layers=3,
                    commitment_weight=0.25,
                    decay=vq_decay,
                    eps=vq_eps,
                    threshold_ema_dead_code=vq_dead_code_threshold,
                )
            )
            self.lins.append(nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim))
            if self.pre_ln:
                self.pre_lns.append(nn.LayerNorm(self.hidden_dim))

        self.lin_in = nn.Linear(in_features=in_channels, out_features=self.hidden_dim)
        self.linear_gnn = nn.Linear(in_features=self.hidden_dim, out_features=local_layers * 3)
        self.pred_local = nn.Linear(in_features=self.hidden_dim, out_features=out_channels)
        self.relu = tlx.ReLU()

    def forward(self, x, edge_index):
        x = self.in_drop(x)
        x = self.lin_in(x)
        x = self.dropout(x)

        id_list = []
        total_commit_loss = 0
        x_local = 0

        num_nodes = tlx.get_tensor_shape(x)[0]

        for layer_id, (local_conv, vq) in enumerate(zip(self.local_convs, self.vqs)):
            if self.pre_ln:
                x = self.pre_lns[layer_id](x)

            if self.gnn == 'gat':
                conv_out = local_conv(x, edge_index, num_nodes=num_nodes)
            else:
                conv_out = local_conv(x, edge_index, edge_weight=None, num_nodes=num_nodes)

            x = conv_out + self.lins[layer_id](x)
            x = self.relu(x)
            x = self.dropout(x)
            x_local = x_local + x

            quantized, code_indices, commit_loss = vq(x)
            id_list.append(tlx.stack(code_indices, axis=1))
            total_commit_loss = total_commit_loss + commit_loss

        id_list_concat = tlx.concat(id_list, axis=1)
        gnn_id = self.linear_gnn(x_local)
        logits = self.pred_local(x_local)

        return logits, total_commit_loss, id_list_concat, gnn_id
