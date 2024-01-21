import logging
import os
import os.path as osp
from collections import Counter
from typing import Callable, List, Optional

import numpy as np
import tensorlayerx as tlx

from gammagl.data import (
    Graph,
    InMemoryDataset,
    download_url,
    extract_tar,
)


class Entities(InMemoryDataset):
    r"""The relational entities networks "AIFB", "MUTAG", "BGS" and "AM" from
    the `"Modeling Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper.
    Training and test splits are given by node indices.
    
    Parameters
    ----------
    root: str, optional
        Root directory where the dataset should be saved.
    name: str, optional
        The name of the dataset (:obj:`"AIFB"`,
        :obj:`"MUTAG"`, :obj:`"BGS"`, :obj:`"AM"`).
    hetero: bool, optional
        If set to :obj:`True`, will save the dataset
        as a :class:`~gammagl.data.HeteroGraph` object.
        (default: :obj:`False`)
    transform: callable, optional
        A function/transform that takes in an
        :obj:`gammagl.data.Graph` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform: callable, optional
        A function/transform that takes in
        an :obj:`gammagl.data.Graph` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

    """

    url = 'https://data.dgl.ai/dataset/{}.tgz'

    def __init__(self, root: str = None, name: str = 'aifb', hetero: bool = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, force_reload: bool = False):
        self.name = name.lower()
        self.hetero = hetero
        assert self.name in ['aifb', 'am', 'mutag', 'bgs']
        super().__init__(root, transform, pre_transform, force_reload = force_reload)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def num_relations(self) -> int:
        return int(tlx.reduce_max(self.data.edge_type)) + 1

    @property
    def num_classes(self) -> int:
        return int(tlx.reduce_max(self.data.train_y)) + 1

    @property
    def raw_file_names(self) -> List[str]:
        return [
            f'{self.name}_stripped.nt.gz',
            'completeDataset.tsv',
            'trainingSet.tsv',
            'testSet.tsv',
        ]

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + 'hetero_data.pt' if self.hetero else tlx.BACKEND + 'data.pt'

    def download(self):
        path = download_url(self.url.format(self.name), self.root)
        extract_tar(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        import gzip

        import pandas as pd
        import rdflib as rdf

        graph_file, task_file, train_file, test_file = self.raw_paths

        with hide_stdout():
            g = rdf.Graph()
            with gzip.open(graph_file, 'rb') as f:
                g.parse(file=f, format='nt')

        freq = Counter(g.predicates())

        relations = sorted(set(g.predicates()), key=lambda p: -freq.get(p, 0))
        subjects = set(g.subjects())
        objects = set(g.objects())
        nodes = list(subjects.union(objects))

        N = len(nodes)
        R = 2 * len(relations)

        relations_dict = {rel: i for i, rel in enumerate(relations)}
        nodes_dict = {node: i for i, node in enumerate(nodes)}

        edges = []
        for s, p, o in g.triples((None, None, None)):
            src, dst, rel = nodes_dict[s], nodes_dict[o], relations_dict[p]
            edges.append([src, dst, 2 * rel])
            edges.append([dst, src, 2 * rel + 1])

        edges = np.array(edges).transpose()
        perm = (N * R * edges[0] + R * edges[1] + edges[2]).argsort()
        edges = edges[:, perm]

        edge_index, edge_type = edges[:2], edges[2]

        if self.name == 'am':
            label_header = 'label_cateogory'
            nodes_header = 'proxy'
        elif self.name == 'aifb':
            label_header = 'label_affiliation'
            nodes_header = 'person'
        elif self.name == 'mutag':
            label_header = 'label_mutagenic'
            nodes_header = 'bond'
        elif self.name == 'bgs':
            label_header = 'label_lithogenesis'
            nodes_header = 'rock'

        labels_df = pd.read_csv(task_file, sep='\t')
        labels_set = set(labels_df[label_header].values.tolist())
        labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}
        nodes_dict = {str(key): val for key, val in nodes_dict.items()}

        train_labels_df = pd.read_csv(train_file, sep='\t')
        train_indices, train_labels = [], []
        for nod, lab in zip(train_labels_df[nodes_header].values,
                            train_labels_df[label_header].values):
            train_indices.append(nodes_dict[nod])
            train_labels.append(labels_dict[lab])

        train_idx = np.array(train_indices)
        train_y = np.array(train_labels)

        test_labels_df = pd.read_csv(test_file, sep='\t')
        test_indices, test_labels = [], []
        for nod, lab in zip(test_labels_df[nodes_header].values,
                            test_labels_df[label_header].values):
            test_indices.append(nodes_dict[nod])
            test_labels.append(labels_dict[lab])

        test_idx = np.array(test_indices)
        test_y = np.array(test_labels)

        data = Graph(edge_index=edge_index, edge_type=edge_type, train_idx=train_idx, train_y=train_y,
                     test_idx=test_idx, test_y=test_y, num_nodes=N)

        if self.hetero:
            data = data.to_heterogeneous(node_type_names=['v'])

        self.save_data(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.upper()}{self.__class__.__name__}()'


class hide_stdout(object):
    def __enter__(self):
        self.level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)

    def __exit__(self, *args):
        logging.getLogger().setLevel(self.level)
