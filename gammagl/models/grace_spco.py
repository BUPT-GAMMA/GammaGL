import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.layers.conv import GCNConv


class LogReg(nn.Module):
    def __init__(self, in_channel, n_class):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(in_features=in_channel,
                            out_features=n_class,
                            W_init='xavier_uniform',
                            b_init='zeros')

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Grace_Spco_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, activation,
                 base_model=GCNConv, k=2):
        super(Grace_Spco_Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k - 1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x, edge_index, edge_attr):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index, edge_attr))
        return x


class Grace_Spco_Model(nn.Module):
    def __init__(self, encoder, num_hidden, num_proj_hidden, tau = 0.5):
        super(Grace_Spco_Model, self).__init__()
        self.encoder = encoder
        self.tau = tau

        self.fc1 = nn.Linear(in_features=num_hidden, out_features=num_proj_hidden)
        self.fc2 = nn.Linear(in_features=num_proj_hidden, out_features=num_hidden)

    def forward(self, x, edge_index, edge_attr):
        return self.encoder(x, edge_index, edge_attr)

    def projection(self, z):
        z = tlx.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1, z2):
        z1 = tlx.l2_normalize(z1, axis=1)
        z2 = tlx.l2_normalize(z2, axis=1)
        return tlx.matmul(z1, tlx.transpose(z2))

    def semi_loss(self, z1, z2):
        f = lambda x: tlx.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -tlx.log(
            tlx.diag(between_sim)
            / (tlx.reduce_sum(refl_sim, axis=1) + tlx.reduce_sum(between_sim, axis=1) - tlx.diag(refl_sim)))

    def batched_semi_loss(self, z1, z2, batch_size):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        num_nodes = tlx.get_tensor_shape(z1)[0]
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: tlx.exp(x / self.tau)
        losses = []

        for i in range(num_batches):
            mask = tlx.arange(i * batch_size, (i + 1) * batch_size if (i + 1) * batch_size <= num_nodes else num_nodes, dtype=tlx.int64)
            refl_sim = f(self.sim(tlx.gather(z1, mask), z1))  # [B, N]
            between_sim = f(self.sim(tlx.gather(z1, mask), z2))  # [B, N]

            losses.append(-tlx.log(
                tlx.diag(between_sim[:, i * batch_size:(i + 1) * batch_size])
                / (tlx.reduce_sum(refl_sim, axis=1) + tlx.reduce_sum(between_sim, axis=1)
                   - tlx.diag(refl_sim[:, i * batch_size:(i + 1) * batch_size]))))

        return tlx.concat(losses)

    def loss(self, z1, z2, mean = True, batch_size = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = tlx.reduce_mean(ret) if mean else tlx.reduce_sum(ret)

        return ret


