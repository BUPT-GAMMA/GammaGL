import numpy as np
import scipy.sparse as sp
import tensorlayerx as tlx
import os
import pandas as pd
from typing import Callable, List, Optional
from gammagl.data import download_url, InMemoryDataset, Graph, extract_tar


class PolBlogs(InMemoryDataset):
    r"""The Political Blogs dataset from the `"The Political Blogosphere and
    the 2004 US Election: Divided they Blog"
    <https://dl.acm.org/doi/10.1145/1134271.1134277>`_ paper.

    :class:`Polblogs` is a graph with 1,490 vertices (representing political
    blogs) and 19,025 edges (links between blogs).
    The links are automatically extracted from a crawl of the front page of the
    blog.
    Each vertex receives a label indicating the political leaning of the blog:
    liberal or conservative.

    Parameters
    ----------
    root: str, optional
        Root directory where the dataset should be saved.
    transform: callable, optional
        A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform: callable, optional
        A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)


    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 1,490
          - 19,025
          - 0
          - 2
    """

    url = 'https://netset.telecom-paris.fr/datasets/polblogs.tar.gz'

    def __init__(self, root: str = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['adjacency.tsv', 'labels.tsv']

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + '_data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_tar(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        edge_index = pd.read_csv(self.raw_paths[0], header=None, sep='\t',
                                 usecols=[0, 1])
        edge_index = np.ascontiguousarray(np.array(edge_index, dtype=np.int64).T)
        adj = sp.csr_matrix(np.eye(1490))
        x = adj.todense()
        y = pd.read_csv(self.raw_paths[1], header=None, sep='\t')
        y = tlx.reshape(tlx.convert_to_tensor(y.values), (-1,))

        data = Graph(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self.save_data(self.collate([data]), self.processed_paths[0])
