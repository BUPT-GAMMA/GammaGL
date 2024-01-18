import numpy as np
from typing import List
from gammagl.data import download_url
from gammagl.utils import read_embeddings


class CA_GrQc():
    r"""The CA-GrQc datasets used in the `"GraphGAN: Graph Representation Learning with Generative Adversarial Nets"
    <https://arxiv.org/pdf/1711.08267.pdf>`_ paper. arXiv-GrQc is from arXiv and covers scientific collaborations
    between authors with papers submitted to the General Relativity and Quantum Cosmology categories. This graph has
    5,242 vertices and 14,496 edges.

    Parameters
    ----------
    dir: str
        Root directory where the dataset should be saved.
    n_emb: int
        Dimension of node embeddings

    """

    url = 'https://raw.githubusercontent.com/hwwang55/GraphGAN/master'

    def __init__(self, dir: str, n_emb: int):

        self.download(dir)
        self.n_node, self.graph = self.read_edges(f'{dir}/CA-GrQc_train.txt', f'{dir}/CA-GrQc_test.txt')

        self.test_edges = self.read_edges_from_file(f'{dir}/CA-GrQc_test.txt')
        self.test_edges_neg = self.read_edges_from_file(f'{dir}/CA-GrQc_test_neg.txt')

        filename=f'{dir}/CA-GrQc_pre_train.emb'

        with open(filename, "r") as f:
            lines = f.readlines()[1:]
            embedding_matrix_d = np.random.rand(self.n_node, n_emb)
            for line in lines:
                emd = line.split()
                embedding_matrix_d[int(emd[0]), :] = [float(item) for item in emd[1:]]

        embedding_matrix_g = embedding_matrix_d.copy()
            
        self.node_embed_init_d = embedding_matrix_d
        self.node_embed_init_g = embedding_matrix_g


    @property
    def file_names(self) -> List[str]:
        names = ['data/link_prediction/CA-GrQc_train.txt', 'data/link_prediction/CA-GrQc_test.txt',
                 'data/link_prediction/CA-GrQc_test_neg.txt', 'pre_train/link_prediction/CA-GrQc_pre_train.emb']
        return [f'{name}' for name in names]

    def download(self, dir):
        for name in self.file_names:
            download_url(f'{self.url}/{name}', dir)

    def read_edges(self, train_filename, test_filename):
        """read data from downloaded files

        Parameters
        ----------
        train_filename: 
            training file name
        test_filename: 
            test file name

        Returns
        -------
        (:obj:`int`, :obj:`dict`): number of nodes in the graph and node_id -> list of neighbors in the graph

        """
        graph = {}
        nodes = set()
        train_edges = self.read_edges_from_file(train_filename)
        test_edges = self.read_edges_from_file(test_filename) if test_filename != "" else []

        for edge in train_edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
            if graph.get(edge[0]) is None:
                graph[edge[0]] = []
            if graph.get(edge[1]) is None:
                graph[edge[1]] = []
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])

        for edge in test_edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
            if graph.get(edge[0]) is None:
                graph[edge[0]] = []
            if graph.get(edge[1]) is None:
                graph[edge[1]] = []

        return len(nodes), graph

    def str_list_to_int(self, str_list):
        return [int(item) for item in str_list]

    def read_edges_from_file(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()
            edges = [self.str_list_to_int(line.split()) for line in lines]
        return edges
