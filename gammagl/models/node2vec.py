import tensorlayerx as tlx
from collections import defaultdict
import numpy as np
from ..utils.num_nodes import maybe_num_nodes
from gammagl.loader import RandomWalk

random_walk = RandomWalk("node2vec")

EPS = 1e-15


class Node2vecModel(tlx.nn.Module):
    r"""The Node2Vec model from the
        `"node2vec: Scalable Feature Learning for Networks"
        <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
        length :obj:`walk_length` are sampled in a given graph, and node embeddings
        are learned via negative sampling optimization.

        Parameters
        ----------
        edge_index: Iterable
            The edge indices.
        edge_weight: Iterable
            The edge weight.
        embedding_dim: int
            The size of each embedding vector.
        walk_length: int
            The walk length.
        p: float
            Likelihood of immediately revisiting a node in the walk.
        q: float
            Control parameter to interpolate between breadth-first strategy and depth-first strategy.
        num_walks: int, optional
            The number of walks to sample for each node.
        window_size: int, optional
            The actual context size which is considered for
            positive samples. This parameter increases the effective sampling
            rate by reusing samples across different source nodes.
        num_negatives: int, optional
            The number of negative samples to use for each positive sample.
        num_nodes: int, optional
            The number of nodes.
        name: str, optional
            model name.

    """

    def __init__(
            self,
            edge_index,
            edge_weight,
            embedding_dim,
            walk_length,
            p,
            q,
            num_walks=10,
            window_size=5,
            num_negatives=1,
            num_nodes=None,
            name=None
    ):
        super(Node2vecModel, self).__init__(name=name)

        assert walk_length >= window_size
        self.edge_index = edge_index
        self.edge_weight = edge_weight

        if tlx.BACKEND == 'mindspore':
            self.N = maybe_num_nodes(tlx.convert_to_numpy(edge_index), num_nodes)
        else:
            self.N = maybe_num_nodes(edge_index, num_nodes)
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_negatives = num_negatives

        self.random_walks = random_walk(self.edge_index, self.num_walks, self.walk_length, edge_weight=self.edge_weight,
                                        p=self.p, q=self.q, num_nodes=self.N)

        self.embedding = tlx.nn.Embedding(self.N, embedding_dim)

    def forward(self, edge_index):
        return self.loss(self.pos_sample(), self.neg_sample())

    def pos_sample(self):
        rw = self.random_walks
        rw = np.array(rw)

        walks = []
        num_walks_per_rw = 1 + self.walk_length - self.window_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.window_size])
        walks = tlx.convert_to_tensor(walks)
        return tlx.concat([walks[i] for i in range(len(walks))], axis=0)

    def neg_sample(self):
        rw = np.random.randint(low=0, high=self.N,
                               size=(self.N * self.num_walks * self.num_negatives, self.walk_length))

        rw = np.array(rw)
        walks = []
        num_walks_per_rw = 1 + self.walk_length - self.window_size

        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.window_size])

        walks = tlx.convert_to_tensor(walks)

        return tlx.concat([walks[i] for i in range(len(walks))], axis=0)

    def loss(self, pos_rw, neg_rw):
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

    def campute(self):
        emb = self.embedding.all_weights
        return emb
