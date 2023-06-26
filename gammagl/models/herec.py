import tensorlayerx as tlx
from gammagl.typing import NodeType, EdgeType
from typing import Dict, List, Optional
import numpy as np
from tensorlayerx.dataflow import DataLoader
from gammagl.utils.random_walk_sample import rw_sample_by_edge_index
from .skipgram import SkipGramModel

EPS = 1e-15


class HERec(tlx.nn.Module):
    r"""The HERec model from the
    `"Heterogeneous Information Network Embedding for Recommendation"
    <https://arxiv.org/pdf/1711.10730.pdf>`_ paper design a meta-path
    based random walk strategy to generate meaningful node sequences
    for network embedding.

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
        walks_per_node: int
            The number of walks to sample for each node.
        num_negative_samples: int
            The number of negative samples to use for each positive sample.
        num_nodes_dict: Dict
            Dictionary holding the number of nodes for each node type.
        target_type: str
            The node type with the label.
        dataset: str
            The name of dataset.
        name: str
            model name
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
            target_type=None,
            dataset=None,
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
        self.target_type = target_type
        self.dataset = dataset

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

        self.edge_index_big = [[], []]
        edge_dict = {}
        self.edge_index_dict_csr = {}
        for keys, edge_index in edge_index_dict.items():
            edge_dict[keys] = edge_index

        self.edge_dict = edge_dict

        self.skipgram = SkipGramModel(embedding_dim=self.embedding_dim, window_size=self.context_size, num_nodes=self.num_nodes_dict[self.target_type])
        self.dummy_idx = count

    def forward(self, pos_rw, neg_rw):
        return self.loss(pos_rw, neg_rw)

    def loader(self, **kwargs):
        r"""Returns the data loader that creates both positive and negative
        random walks on the heterogeneous graph."""
        return DataLoader(range(self.num_nodes_dict[self.metapath[0][0]]), collate_fn=self._sample, **kwargs)

    def campute(self, batch=None):
        r"""Returns the embeddings for the nodes in :obj:`batch` of type
        :obj:`node_type`."""
        emb = self.skipgram.embedding.all_weights[0]
        return emb if batch is None else tlx.gather(emb, batch, axis=0)

    def _sample(self, batch):
        batch = tlx.convert_to_tensor(batch, dtype=tlx.int64)
        return self._pos_sample(batch), self._neg_sample(batch)

    def _pos_sample(self, batch):
        batch = tlx.convert_to_numpy(batch)
        batch = [it for i in range(self.walks_per_node) for it in batch]

        if self.metapath[0][0] == self.target_type:
            rws = [tlx.convert_to_tensor(batch, dtype=tlx.int64)]
        else:
            rws = []
        for i in range(self.walk_length - 1):
            keys = self.metapath[i % len(self.metapath)]
            batch = rw_sample_by_edge_index(self.edge_dict[keys], batch, 2)
            batch = list(map(lambda x: x[-1], batch))
            if keys[-1] == self.target_type:
                rws.append(tlx.convert_to_tensor(batch, dtype=tlx.int64))

        rw = tlx.stack(rws, axis=-1)

        walks = []
        num_walks_per_rw = 1 + len(rw[0]) - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return tlx.concat(walks, axis=0)

    def _neg_sample(self, batch):
        length = self.walks_per_node * self.num_negative_samples
        rws = []
        for i in range(self.walk_length):
            batch = np.random.randint(low=0, high=self.num_nodes_dict[self.target_type], size=(length,))
            batch = tlx.convert_to_tensor(batch, dtype=tlx.int64)
            rws.append(batch)
        rw = tlx.stack(rws, axis=-1)

        walks = []
        num_walks_per_rw = 1 + len(rw[0]) - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return tlx.concat(walks, axis=0)

    def loss(self, pos_rw, neg_rw):
        r"""Computes the loss given positive and negative random walks."""
        return self.skipgram(pos_rw, neg_rw)
