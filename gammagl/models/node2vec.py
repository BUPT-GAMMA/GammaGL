import tensorlayerx as tlx
from collections import defaultdict
import numpy as np
from ..utils.num_nodes import maybe_num_nodes

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
        num_walks: int
            The number of walks to sample for each node.
        window_size: int
            The actual context size which is considered for
            positive samples. This parameter increases the effective sampling
            rate by reusing samples across different source nodes.
        num_negatives: int
            The number of negative samples to use for each positive sample.
        num_nodes: int
            The number of nodes.
        name: str
            model name
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

        self.N = maybe_num_nodes(edge_index, num_nodes)
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_negatives = num_negatives

        self.random_walks = self.generate_random_walks()

        self.embedding = tlx.nn.Embedding(self.N, embedding_dim)

    def forward(self, edge_index):
        return self.loss(self.pos_sample(), self.neg_samples())

    def pos_sample(self):
        rw = self.random_walks
        rw = np.array(rw)

        walks = []
        num_walks_per_rw = 1 + self.walk_length - self.window_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.window_size])
        walks = tlx.convert_to_tensor(walks)
        return tlx.concat([walks[i] for i in range(len(walks))], axis=0)

    def neg_samples(self):
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
        rest = tlx.convert_to_tensor(np.array(pos_rw[:, 1:]))

        h_start = tlx.reshape(self.embedding(start), (pos_rw.shape[0], 1, self.embedding_dim))
        h_rest = tlx.reshape(self.embedding(tlx.reshape(rest, (-1, 1))), (pos_rw.shape[0], -1, self.embedding_dim))

        out = tlx.reshape(tlx.ops.reduce_sum((h_start * h_rest), axis=-1), (-1, 1))

        pos_loss = -tlx.ops.reduce_mean(tlx.log(tlx.sigmoid(out) + EPS))

        # Negative loss.
        start = neg_rw[:, 0]
        rest = tlx.convert_to_tensor(np.array(neg_rw[:, 1:]))

        h_start = tlx.reshape(self.embedding(start), (neg_rw.shape[0], 1, self.embedding_dim))
        h_rest = tlx.reshape(self.embedding(tlx.reshape(rest, (-1, 1))), (neg_rw.shape[0], -1, self.embedding_dim))

        out = tlx.reshape(tlx.ops.reduce_sum((h_start * h_rest), axis=-1), (-1, 1))

        neg_loss = -tlx.ops.reduce_mean(tlx.log(1 - tlx.sigmoid(out) + EPS))

        return pos_loss + neg_loss

    def campute(self):
        emb = self.embedding.all_weights
        return emb

    def get_neighbors(self):
        edge_index = self.edge_index
        src, dst = edge_index[0], edge_index[1]
        src = np.array(src)
        dst = np.array(dst)
        node_neighbor = {}
        self.nodes_weight = {}
        index = 0
        for src_node in src:
            if src_node not in node_neighbor.keys():
                node_neighbor[src_node] = list()

            node_neighbor[src_node].append(dst[index])
            self.nodes_weight[(src_node, dst[index])] = self.edge_weight[index]
            index += 1

        return node_neighbor

    def compute_probabilities(self):
        neighbor = self.neighbor

        src = self.edge_index[0]
        src = tlx.convert_to_numpy(src)
        node = set(src)

        for source_node in node:
            for current_node in neighbor[source_node]:
                probs_ = list()
                for destination in neighbor[current_node]:
                    weight = tlx.convert_to_numpy(self.nodes_weight[(current_node, destination)])
                    if source_node == destination:
                        prob_ = (1 / self.p) * weight
                    elif destination in neighbor[source_node]:
                        prob_ = 1 * weight
                    else:
                        prob_ = (1 / self.q) * weight

                    probs_.append(prob_)

                self.probs[source_node]['probabilities'][current_node] = probs_ / np.sum(probs_)

    def generate_random_walks(self):
        self.neighbor = self.get_neighbors()
        neighbor = self.neighbor

        self.probs = defaultdict(dict)
        for node in range(self.N):
            self.probs[node]['probabilities'] = dict()
        self.compute_probabilities()

        src = self.edge_index[0]
        src = tlx.convert_to_numpy(src)
        node = set(src)

        walks = list()
        for start_node in node:
            for i in range(self.num_walks):

                walk = [int(start_node)]
                walk_options = neighbor[walk[-1]]
                if len(walk_options) == 0:
                    break
                first_step = np.random.choice(walk_options)
                walk.append(int(first_step))

                for k in range(self.walk_length - 2):
                    walk_options = neighbor[walk[-1]]
                    if len(walk_options) == 0:
                        break
                    probabilities = self.probs[walk[-2]]['probabilities'][walk[-1]]
                    if tlx.BACKEND == 'paddle':
                        probabilities = np.concatenate(probabilities, axis=0)
                    next_step = np.random.choice(walk_options, p=probabilities)
                    walk.append(int(next_step))

                walks.append(walk)

        np.random.shuffle(walks)

        return walks
