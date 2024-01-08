import tensorlayerx as tlx
from tensorlayerx.nn import Embedding
from gammagl.typing import NodeType, EdgeType
from typing import Dict, List, Optional, Tuple
from tensorlayerx.dataflow import DataLoader
from gammagl.utils.random_walk_sample import rw_sample_by_edge_index
import numpy as np
from gammagl.data import Graph
from .skipgram import SkipGramModel

EPS = 1e-15


class MetaPath2Vec(tlx.nn.Module):
    r"""The MetaPath2Vec model from the `"metapath2vec: Scalable Representation
        Learning for Heterogeneous Networks"
        <https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf>`_ paper
        where random walks based on a given :obj:`metapath` are sampled in a heterogeneous graph,
        and node embeddings are learned via negative sampling optimization.

        Parameters
        ----------
        edge_index_dict: Dict
            Dictionary holding edge indices for each
            :obj:`(src_node_type, rel_type, dst_node_type)`
            edge type present in the heterogeneous graph.
        embedding_dim: int
            The size of each embedding vector.
        metapath: List
            The metapath described as a list
            of :obj:`(src_node_type, rel_type, dst_node_type)` tuples.
        walk_length: int
            The walk length.
        context_size: int
            The actual context size which is considered for
            positive samples. This parameter increases the effective sampling
            rate by reusing samples across different source nodes.
        walks_per_node: int, optional
            The number of walks to sample for each node.
        num_negative_samples: int, optional
            The number of negative samples to use for each positive sample.
        num_nodes_dict: Dict, optional
            Dictionary holding the number of nodes for each node type.
        name: str, optional
            model name.

    """

    def __init__(
            self,
            edge_index_dict,
            embedding_dim: int,
            metapath: List[EdgeType],
            walk_length: int,
            context_size: int,
            walks_per_node: int = 1,
            num_negative_samples: int = 1,
            num_nodes_dict: Optional[Dict[NodeType, int]] = None,
            name=None
    ):
        super().__init__(name=name)

        if num_nodes_dict is None:
            num_nodes_dict = {}
            for keys, edge_index in edge_index_dict.items():
                edge_index = tlx.convert_to_tensor(edge_index)
                key = keys[0]
                N = int(tlx.ops.reduce_max(edge_index[0], axis=0, keepdims=False) + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

                key = keys[-1]
                N = int(tlx.ops.reduce_max(edge_index[1], axis=0, keepdims=False) + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

        assert walk_length + 1 >= context_size
        if walk_length > len(metapath) and metapath[0][0] != metapath[-1][-1]:
            raise AttributeError(
                "The 'walk_length' is longer than the given 'metapath', but "
                "the 'metapath' does not denote a cycle")

        self.embedding_dim = embedding_dim
        self.metapath = metapath
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.num_nodes_dict = num_nodes_dict

        types = set([x[0] for x in metapath]) | set([x[-1] for x in metapath])
        types = sorted(list(types))

        count = 0
        self.start, self.end = {}, {}
        for key in types:
            self.start[key] = count
            count += num_nodes_dict[key]
            self.end[key] = count

        offset = [self.start[metapath[0][0]]]
        offset += [self.start[keys[-1]] for keys in metapath] * int((walk_length / len(metapath)) + 1)
        offset = offset[:walk_length]
        assert len(offset) == walk_length
        self.offset = tlx.convert_to_tensor(offset, dtype=tlx.int64)

        edge_dict = {}
        for keys, edge_index in edge_index_dict.items():
            edge_dict[keys] = edge_index

        self.edge_dict = edge_dict

        # + 1 denotes a dummy node used to link to for isolated nodes.
        self.skipgram = SkipGramModel(embedding_dim=self.embedding_dim, window_size=self.context_size, num_nodes=count)
        self.dummy_idx = count

    def forward(self, pos_rw, neg_rw):
        return self.loss(pos_rw, neg_rw)

    def loader(self, **kwargs):
        r"""Returns the data loader that creates both positive and negative
        random walks on the heterogeneous graph."""
        return DataLoader(dataset=range(self.num_nodes_dict[self.metapath[0][0]]), collate_fn=self._sample, **kwargs)

    def campute(self, node_type, batch=None):
        r"""Returns the embeddings for the nodes in :obj:`batch` of type
        :obj:`node_type`."""
        emb = self.skipgram.embedding.all_weights[0][self.start[node_type]:self.end[node_type]]
        return emb if batch is None else tlx.gather(emb, batch, axis=0)

    def _sample(self, batch):
        batch = tlx.convert_to_tensor(batch, dtype=tlx.int64)
        return self._pos_sample(batch), self._neg_sample(batch)

    def _pos_sample(self, batch):
        # batch = tlx.convert_to_tensor(batch, dtype=tlx.int64)
        batch = tlx.convert_to_numpy(batch)
        batch = [it for i in range(self.walks_per_node) for it in batch]

        rws = [tlx.convert_to_tensor(batch, dtype=tlx.int64)]
        for i in range(self.walk_length - 1):
            keys = self.metapath[i % len(self.metapath)]
            batch = rw_sample_by_edge_index(self.edge_dict[keys], batch, 2)
            batch = list(map(lambda x: x[-1], batch))
            rws.append(tlx.convert_to_tensor(batch, dtype=tlx.int64))

        rw = tlx.stack(rws, axis=-1)
        rw = tlx.add(rw, tlx.reshape(self.offset, (1, -1)))

        walks = []
        num_walks_per_rw = 1 + self.walk_length - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return tlx.concat(walks, axis=0)

    def _neg_sample(self, batch):
        batch = tlx.convert_to_numpy(batch)
        batch = [it for i in range(self.walks_per_node * self.num_negative_samples) for it in batch]

        rws = [tlx.convert_to_tensor(batch, dtype=tlx.int64)]
        for i in range(self.walk_length - 1):
            keys = self.metapath[i % len(self.metapath)]
            batch = np.random.randint(low=0, high=self.num_nodes_dict[keys[-1]], size=(len(batch),))
            batch = tlx.convert_to_tensor(batch, dtype=tlx.int64)
            rws.append(batch)

        rw = tlx.stack(rws, axis=-1)
        rw = tlx.add(rw, tlx.reshape(self.offset, (1, -1)))

        walks = []
        num_walks_per_rw = 1 + self.walk_length - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return tlx.concat(walks, axis=0)

    def loss(self, pos_rw, neg_rw):
        r"""Computes the loss given positive and negative random walks."""

        return self.skipgram(pos_rw, neg_rw)
