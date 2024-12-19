import tensorlayerx as tlx

EPS = 1e-15


class SkipGramModel(tlx.nn.Module):
    def __init__(
            self,
            embedding_dim,
            window_size=5,
            num_nodes=None,
            name=None
    ):
        super(SkipGramModel, self).__init__(name=name)

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.window_size = window_size

        self.embedding = tlx.nn.Embedding(self.num_nodes, embedding_dim)

    def forward(self, pos_rw, neg_rw):
        # Positive loss.
        start = pos_rw[:, 0]
        rest = tlx.convert_to_tensor(tlx.convert_to_numpy(pos_rw[:, 1:]))

        h_start = tlx.reshape(self.embedding(start), (pos_rw.shape[0], 1, self.embedding_dim))
        h_rest = tlx.reshape(self.embedding(tlx.reshape(rest, (-1, 1))), (pos_rw.shape[0], -1, self.embedding_dim))

        out = tlx.reshape(tlx.ops.reduce_sum((h_start * h_rest), axis=-1), (-1, 1))

        pos_loss = -tlx.ops.reduce_mean(tlx.log(tlx.sigmoid(out) + EPS))

        # Negative loss.
        start = neg_rw[:, 0]
        rest = tlx.convert_to_tensor(tlx.convert_to_numpy(neg_rw[:, 1:]))

        h_start = tlx.reshape(self.embedding(start), (neg_rw.shape[0], 1, self.embedding_dim))
        h_rest = tlx.reshape(self.embedding(tlx.reshape(rest, (-1, 1))), (neg_rw.shape[0], -1, self.embedding_dim))

        out = tlx.reshape(tlx.ops.reduce_sum((h_start * h_rest), axis=-1), (-1, 1))

        neg_loss = -tlx.ops.reduce_mean(tlx.log(1 - tlx.sigmoid(out) + EPS))

        return pos_loss + neg_loss
