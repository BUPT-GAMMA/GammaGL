import os
import os.path as osp
os.environ['TL_BACKEND'] = 'tensorflow'# set your backend here, default `tensorflow`, you can choose 'paddle'、'tensorflow'、'torch'
import sys
sys.path.append(os.getcwd())
from typing import Callable, Optional
import tensorlayerx as tlx
from gammagl.data import InMemoryDataset,download_url,Graph
import pickle
import numpy as np
import scipy.sparse as sp
from gammagl.utils.loop import remove_self_loops
from gammagl.utils.undirected import to_undirected


class Coauthor(InMemoryDataset):
    r"""The Coauthor CS and Coauthor Physics networks from the
    `"Pitfalls of Graph Neural Network Evaluation"
    <https://arxiv.org/abs/1811.05868>`_ paper.
    Nodes represent authors that are connected by an edge if they co-authored a
    paper.
    Given paper keywords for each author's papers, the task is to map authors
    to their respective field of study.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"CS"`,
            :obj:`"Physics"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    Stats:
        .. list-table::
            :widths: 10 10 10 10 10
            :header-rows: 1

            * - Name
              - #nodes
              - #edges
              - #features
              - #classes
            * - CS
              - 18,333
              - 163,788
              - 6,805
              - 15
            * - Physics
              - 34,493
              - 495,924
              - 8,415
              - 5
    """

    url = 'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/'

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        assert name.lower() in ['cs', 'physics']
        self.name = 'CS' if name.lower() == 'cs' else 'Physics'
        super().__init__(root, transform, pre_transform)
        with open(self.processed_paths[0], 'rb') as f:
            self.data, self.slices = pickle.load(f)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'ms_academic_{self.name[:3].lower()}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url + self.raw_file_names, self.raw_dir)

    def process(self):
        data = np.load(self.raw_paths[0])
        x = sp.csr_matrix((data['attr_data'], data['attr_indices'], data['attr_indptr']),
                      data['attr_shape']).todense()
        x = np.array(x)
        x[x > 0] = 1
        x=tlx.convert_to_tensor(x,dtype=tlx.float32)

        adj = sp.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']),
                            data['adj_shape']).tocoo()
        edge_index = np.array([adj.row, adj.col])
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = tlx.convert_to_tensor(edge_index, dtype=tlx.int32)

        edge_index = to_undirected(edge_index, num_nodes=x.shape[0])

 
        y = tlx.convert_to_tensor(data['labels'], dtype=tlx.int32)

        data = Graph(x=x, edge_index=edge_index, y=y)

        data = data if self.pre_transform is None else self.pre_transform(data)
        #data = self.pre_transform(data)
        data, slices = self.collate([data])
        with open(self.processed_paths[0], 'wb') as f:
            pickle.dump((data,slices), f)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.name}()'

#data=Coauthor(root='./Coauthor/',name='cs')
#data.process()
#print(data[0].x)
#print(data[0].edge_index.shape)

