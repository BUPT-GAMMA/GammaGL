import os
import os.path as osp
import pickle
import shutil

import tensorlayerx as tlx
from tensorlayerx import convert_to_tensor
from tqdm import tqdm

from gammagl.data import (
    Graph,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class ZINC(InMemoryDataset):
    r"""The ZINC dataset from the `ZINC database
    <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559>`_ and the
    `"Automatic Chemical Design Using a Data-Driven Continuous Representation
    of Molecules" <https://arxiv.org/abs/1610.02415>`_ paper, containing about
    250,000 molecular graphs with up to 38 heavy atoms.
    The task is to regress the penalized :obj:`logP` (also called constrained
    solubility in some works), given by :obj:`y = logP - SAS - cycles`, where
    :obj:`logP` is the water-octanol partition coefficient, :obj:`SAS` is the
    synthetic accessibility score, and :obj:`cycles` denotes the number of
    cycles with more than six atoms.
    Penalized :obj:`logP` is a score commonly used for training molecular
    generation models, see, *e.g.*, the
    `"Junction Tree Variational Autoencoder for Molecular Graph Generation"
    <https://proceedings.mlr.press/v80/jin18a.html>`_ and
    `"Grammar Variational Autoencoder"
    <https://proceedings.mlr.press/v70/kusner17a.html>`_ papers.

    Args:
        root (string): Root directory where the dataset should be saved.
        subset (boolean, optional): If set to :obj:`True`, will only load a
            subset of the dataset (12,000 molecular graphs), following the
            `"Benchmarking Graph Neural Networks"
            <https://arxiv.org/abs/2003.00982>`_ paper. (default: :obj:`False`)
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`gammagl.data.Graph` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`gammagl.data.Graph` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`gammagl.data.Graph` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1'
    split_url = ('https://raw.githubusercontent.com/graphdeeplearning/'
                 'benchmarking-gnns/master/data/molecules/{}.index')

    def __init__(self, root: str = None, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = self.load_data(path)

    @property
    def raw_file_names(self):
        return [
            'train.pickle', 'val.pickle', 'test.pickle', 'train.index',
            'val.index', 'test.index'
        ]

    @property
    def processed_dir(self):
        name = 'subset' if self.subset else 'full'
        return osp.join(self.root, name, tlx.BACKEND)

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'molecules'), self.raw_dir)
        os.unlink(path)

        for split in ['train', 'val', 'test']:
            download_url(self.split_url.format(split), self.raw_dir)

    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                mols = pickle.load(f)
            indices = range(len(mols))
            if self.subset:
                with open(osp.join(self.raw_dir, f'{split}.index'), 'r') as f:
                    indices = [int(x) for x in f.read()[:-1].split(',')]

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')
            data_list = []
            for idx in indices:
                mol = mols[idx]
                x = convert_to_tensor(value=mol['atom_type'].view(-1, 1).numpy(), dtype=tlx.int64)
                y = convert_to_tensor(value=mol['logP_SA_cycle_normalized'].numpy(), dtype=tlx.float32)

                adj = mol['bond_type']
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                edge_attr = convert_to_tensor(value=adj[edge_index[0], edge_index[1]].numpy(), dtype=tlx.int64)
                edge_index = convert_to_tensor(value=edge_index.numpy(), dtype=tlx.int64)

                data = Graph(x=x, edge_index=edge_index, edge_attr=edge_attr,
                             y=y, to_tensor=True)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
                pbar.update(1)
            pbar.close()

            self.data, self.slices = self.collate(data_list)
            self.save_data((self.data, self.slices), osp.join(self.processed_dir, f'{split}.pt'))
