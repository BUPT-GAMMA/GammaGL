"""Models used by GP2F: Graph Prompting for Few-shot Graph Transfer."""

import tensorlayerx as tlx


class GP2FGraphConv(tlx.nn.Module):

    def __init__(self, in_channels, out_channels, name=None):
        super().__init__(name=name)
        self.linear = tlx.nn.Linear(in_features=in_channels,
                                    out_features=out_channels,
                                    W_init="xavier_uniform", b_init=None)
        self.bias = self._get_weights(
            "bias", shape=(1, out_channels), init=tlx.initializers.Zeros())

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        if num_nodes is None:
            num_nodes = tlx.get_tensor_shape(x)[0]
        if edge_weight is None:
            edge_weight = tlx.ones(shape=(tlx.get_tensor_shape(edge_index)[1],))
        edge_weight = tlx.reshape(edge_weight, (-1,))
        src, dst = edge_index[0], edge_index[1]

        deg = tlx.reshape(tlx.unsorted_segment_sum(edge_weight, dst, num_nodes), (-1,))
        deg_inv_sqrt = tlx.pow(deg, -0.5)
        deg_inv_sqrt = tlx.where(tlx.is_inf(deg_inv_sqrt),
                                 tlx.zeros_like(deg_inv_sqrt), deg_inv_sqrt)
        norm = (tlx.gather(deg_inv_sqrt, src) * edge_weight *
                tlx.gather(deg_inv_sqrt, dst))

        x = self.linear(x)
        messages = tlx.gather(x, src) * tlx.expand_dims(norm, axis=-1)
        return tlx.unsorted_segment_sum(messages, dst, num_nodes) + self.bias


class GP2FEncoder(tlx.nn.Module):
    r"""Two-layer PReLU GCN used by the released GP2F GRACE implementation."""

    def __init__(self, in_channels, hidden_channels=128, num_layers=2,
                 dropout=0.0, name=None):
        super().__init__(name=name)
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = tlx.nn.PRelu()
        self.dropout_layer = tlx.nn.Dropout(p=dropout)
        self.convs = tlx.nn.ModuleList()
        self.convs.append(GP2FGraphConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GP2FGraphConv(hidden_channels, hidden_channels))

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight, num_nodes)
            if i < self.num_layers - 1:
                x = self.act(x)
                if self.dropout > 0:
                    x = self.dropout_layer(x)
        return x


class GP2FGRACE(tlx.nn.Module):
    """GRACE wrapper used for Cora pretraining of the GP2F source encoder."""

    def __init__(self, in_channels, hidden_channels=128, num_layers=2,
                 temperature=0.5, name=None):
        super().__init__(name=name)
        self.encoder = GP2FEncoder(in_channels, hidden_channels, num_layers)
        self.fc1 = tlx.nn.Linear(in_features=hidden_channels,
                                 out_features=hidden_channels)
        self.fc2 = tlx.nn.Linear(in_features=hidden_channels,
                                 out_features=hidden_channels)
        self.temperature = temperature

    def encode(self, x, edge_index, edge_weight=None, num_nodes=None):
        return self.encoder(x, edge_index, edge_weight, num_nodes)

    def project(self, x):
        return self.fc2(tlx.elu(self.fc1(x)))

    @staticmethod
    def similarity(x, y):
        x = x / tlx.sqrt(tlx.maximum(
            tlx.reduce_sum(tlx.square(x), axis=1, keepdims=True),
            tlx.convert_to_tensor(1e-12)))
        y = y / tlx.sqrt(tlx.maximum(
            tlx.reduce_sum(tlx.square(y), axis=1, keepdims=True),
            tlx.convert_to_tensor(1e-12)))
        return tlx.matmul(x, tlx.transpose(y))

    def _semi_loss(self, x, y):
        refl = tlx.exp(self.similarity(x, x) / self.temperature)
        between = tlx.exp(self.similarity(x, y) / self.temperature)
        return -tlx.log(tlx.diag(between, 0) / (
            tlx.reduce_sum(refl, axis=1) + tlx.reduce_sum(between, axis=1)
            - tlx.diag(refl, 0) + 1e-12))

    def loss(self, z1, z2):
        z1, z2 = self.project(z1), self.project(z2)
        return tlx.reduce_mean((self._semi_loss(z1, z2) +
                                self._semi_loss(z2, z1)) * 0.5)


class GP2FFeatureProjector(tlx.nn.Module):
    """Projects target features into the source graph feature space."""

    def __init__(self, in_channels, out_channels, name=None):
        super().__init__(name=name)
        self.linear = tlx.nn.Linear(in_features=in_channels,
                                    out_features=out_channels)
        self.act = tlx.nn.PRelu()

    def forward(self, x):
        return self.act(self.linear(x))


class GP2FAdapter(tlx.nn.Module):
    def __init__(self, hidden_channels, bottleneck_channels, name=None):
        super().__init__(name=name)
        self.down = tlx.nn.Linear(in_features=hidden_channels,
                                  out_features=bottleneck_channels)
        self.up = tlx.nn.Linear(in_features=bottleneck_channels,
                                out_features=hidden_channels)

    def forward(self, x):
        return self.up(tlx.relu(self.down(x)))


class GP2FPrompt(tlx.nn.Module):
    """Layer-wise adapters used with a frozen :class:`GP2FEncoder`."""

    def __init__(self, hidden_channels, num_layers=2, bottleneck_channels=32,
                 alpha_init=0.1, name=None):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.adapters = tlx.nn.ModuleList()
        self.alphas = []
        for i in range(num_layers):
            self.adapters.append(GP2FAdapter(hidden_channels, bottleneck_channels))
            self.alphas.append(self._get_weights(
                "alpha_{}".format(i), shape=(1,),
                init=tlx.initializers.Constant(value=alpha_init)))

    def forward(self, x, edge_index, encoder, edge_weight=None, num_nodes=None):
        for i, (conv, adapter) in enumerate(zip(encoder.convs, self.adapters)):
            x_gnn = conv(x, edge_index, edge_weight, num_nodes)
            x = x_gnn + self.alphas[i] * adapter(x_gnn)
            if i < encoder.num_layers - 1:
                x = encoder.act(x)
        return x


class GP2FDualBranch(tlx.nn.Module):
    """Unconstrained adaptive fusion of the frozen and prompted branches."""

    def __init__(self, alpha_init=0.5, name=None):
        super().__init__(name=name)
        self.alpha = self._get_weights(
            "alpha", shape=(1,),
            init=tlx.initializers.Constant(value=alpha_init))

    def forward(self, frozen_output, prompted_output, frozen_similarity=None,
                prompted_similarity=None):
        output = self.alpha * frozen_output + (1.0 - self.alpha) * prompted_output
        if frozen_similarity is None or prompted_similarity is None:
            return output
        similarity = (self.alpha * frozen_similarity +
                      (1.0 - self.alpha) * prompted_similarity)
        return similarity, output


class GP2FClassifier(tlx.nn.Module):
    def __init__(self, in_channels, out_channels, name=None):
        super().__init__(name=name)
        self.linear = tlx.nn.Linear(in_features=in_channels,
                                    out_features=out_channels,
                                    W_init="xavier_uniform")

    def forward(self, x):
        return self.linear(x)


class GP2FModel(tlx.nn.Module):
    """Inference wrapper for the GP2F target-domain model.

    The source encoder is expected to be loaded from a GRACE checkpoint and
    frozen by the training script. Keeping that policy outside this wrapper
    preserves the released GP2F optimization procedure.
    """

    def __init__(self, source_feature_dim, target_feature_dim, hidden_channels,
                 out_channels, encoder, num_layers=2, bottleneck_channels=32,
                 name=None):
        super().__init__(name=name)
        self.encoder = encoder
        self.projector = GP2FFeatureProjector(target_feature_dim,
                                              source_feature_dim)
        self.prompt = GP2FPrompt(hidden_channels, num_layers,
                                 bottleneck_channels)
        self.fusion = GP2FDualBranch()
        self.classifier = GP2FClassifier(hidden_channels, out_channels)

    def encode(self, x, edge_index, edge_weight=None, num_nodes=None):
        features = self.projector(x)
        frozen_output = self.encoder(features, edge_index, edge_weight, num_nodes)
        prompted_output = self.prompt(features, edge_index, self.encoder,
                                      edge_weight, num_nodes)
        return self.fusion(frozen_output, prompted_output)

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        return self.classifier(self.encode(x, edge_index, edge_weight, num_nodes))
