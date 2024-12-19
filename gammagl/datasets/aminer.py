import os
import os.path as osp
import shutil
from typing import Callable, List, Optional

import tensorlayerx as tlx

from ..data import (
    HeteroGraph,
    InMemoryDataset,
    download_url,
    extract_zip
)

from gammagl.utils import coalesce
import numpy as np
import pandas as pd


class AMiner(InMemoryDataset):
    r"""The heterogeneous AMiner dataset from the `"metapath2vec: Scalable
    Representation Learning for Heterogeneous Networks"
    <https://dl.acm.org/doi/pdf/10.1145/3097983.3098036>`_ paper, consisting of nodes from
    type :obj:`"paper"`, :obj:`"author"` and :obj:`"venue"`.
    Aminer is a heterogeneous graph containing three types of entities - author
    (1,693,531 nodes), venue (3,883 nodes), and paper (3,194,405 nodes).
    Venue categories and author research interests are available as ground
    truth labels for a subset of nodes.

    Parameters
    ----------
    root: str, optional
        Root directory where the dataset should be saved.
    transform: callable, optional
        A function/transform that takes in an
        :obj:`gammagl.data.HeteroGraph` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform: callable, optional
        A function/transform that takes in
        an :obj:`gammagl.data.HeteroGraph` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

    """

    url = 'https://www.dropbox.com/s/1bnz8r7mofx0osf/net_aminer.zip?dl=1'
    y_url = 'https://www.dropbox.com/s/nkocx16rpl4ydde/label.zip?dl=1'

    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None,
                 force_reload: bool = False):
        super().__init__(root, transform, pre_transform, pre_filter, force_reload = force_reload)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'id_author.txt', 'id_conf.txt', 'paper.txt', 'paper_author.txt',
            'paper_conf.txt', 'label'
        ]

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + '_data.pt'

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'net_aminer'), self.raw_dir)
        os.unlink(path)
        path = download_url(self.y_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        data = HeteroGraph()

        # Get author labels.
        path = osp.join(self.raw_dir, 'id_author.txt')
        author = pd.read_csv(path, sep='\t', names=['idx', 'name'],
                             index_col=1)

        path = osp.join(self.raw_dir, 'label',
                        'googlescholar.8area.author.label.txt')
        df = pd.read_csv(path, sep=' ', names=['name', 'y'])
        df = df.join(author, on='name')

        data['author'].y = tlx.convert_to_tensor(df['y'].values) - 1
        data['author'].y_index = tlx.convert_to_tensor(df['idx'].values)

        # Get venue labels.
        path = osp.join(self.raw_dir, 'id_conf.txt')
        venue = pd.read_csv(path, sep='\t', names=['idx', 'name'], index_col=1)

        path = osp.join(self.raw_dir, 'label',
                        'googlescholar.8area.venue.label.txt')
        df = pd.read_csv(path, sep=' ', names=['name', 'y'])
        df = df.join(venue, on='name')

        data['venue'].y = tlx.convert_to_tensor(df['y'].values) - 1
        data['venue'].y_index = tlx.convert_to_tensor(df['idx'].values)

        # Get paper<->author connectivity.
        path = osp.join(self.raw_dir, 'paper_author.txt')
        paper_author = pd.read_csv(path, sep='\t', header=None)
        paper_author = tlx.convert_to_tensor(paper_author.values)
        if tlx.BACKEND == 'mindspore':
            paper_author = paper_author.T
        else:
            paper_author = tlx.ops.transpose(paper_author)
        M, N = int(max(paper_author[0]) + 1), int(max(paper_author[1]) + 1)
        paper_author = coalesce(paper_author, num_nodes=max(M, N))
        data['paper'].num_nodes = M
        data['author'].num_nodes = N
        data['paper', 'written_by', 'author'].edge_index = paper_author
        paper_author = tlx.convert_to_numpy(paper_author)
        data['author', 'writes', 'paper'].edge_index = tlx.convert_to_tensor(np.flip(paper_author, axis=0).copy())

        # Get paper<->venue connectivity.
        path = osp.join(self.raw_dir, 'paper_conf.txt')
        paper_venue = pd.read_csv(path, sep='\t', header=None)
        paper_venue = tlx.convert_to_tensor(paper_venue.values)
        if tlx.BACKEND == 'mindspore':
            paper_venue = paper_venue.T
        else:
            paper_venue = tlx.ops.transpose(paper_venue)
        M, N = int(max(paper_venue[0]) + 1), int(max(paper_venue[1]) + 1)
        paper_venue = coalesce(paper_venue, num_nodes=max(M, N))
        data['venue'].num_nodes = N
        data['paper', 'published_in', 'venue'].edge_index = paper_venue
        paper_venue = tlx.convert_to_numpy(paper_venue)
        data['venue', 'publishes', 'paper'].edge_index = tlx.convert_to_tensor(np.flip(paper_venue, axis=0).copy())


        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save_data(self.collate([data]), self.processed_paths[0])
