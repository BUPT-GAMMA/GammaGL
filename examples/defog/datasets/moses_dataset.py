import os
import os.path as osp
import numpy as np
import tensorlayerx as tlx
from typing import Callable, List, Optional

from gammagl.data import (
    Graph,
    InMemoryDataset,
    download_url,
)


class MOSESDataset(InMemoryDataset):
    r"""The MOSES dataset for molecular graph generation.

    From the `"Molecular Sets (MOSES): A Benchmarking Platform for Molecular
    Generation Models" <https://arxiv.org/abs/1811.12823>`_ benchmark.
    Contains ~1.9M drug-like molecules filtered from ZINC Clean Leads.

    Requires **RDKit** (``pip install rdkit``).

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset should be saved.
    split : str
        Which split to load: ``'train'``, ``'val'``, or ``'test'``.
    transform : callable, optional
        A transform applied to each :class:`Graph` on access.
    pre_transform : callable, optional
        A transform applied during processing before saving.
    pre_filter : callable, optional
        A filter applied during processing.
    force_reload : bool
        Whether to re-process the dataset.
    """

    train_url = ('https://media.githubusercontent.com/media/molecularsets/'
                 'moses/master/data/train.csv')
    val_url = ('https://media.githubusercontent.com/media/molecularsets/'
               'moses/master/data/test_scaffolds.csv')
    test_url = ('https://media.githubusercontent.com/media/molecularsets/'
                'moses/master/data/test.csv')

    ATOM_DECODER = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H']
    ATOM_ENCODER = {atom: i for i, atom in enumerate(ATOM_DECODER)}
    NUM_ATOM_TYPES = 8
    NUM_EDGE_TYPES = 5  # no-bond=0, single=1, double=2, triple=3, aromatic=4

    STATS = {
        'valencies': [4, 3, 4, 2, 1, 1, 1, 1],
        'atom_weights': {0: 12, 1: 14, 2: 32, 3: 16, 4: 19, 5: 35.4,
                         6: 79.9, 7: 1},
        'max_weight': 350.0,
        'num_node_types': 8,
        'num_edge_types': 5,
        'n_nodes': np.array([
            0., 0., 0., 0., 0., 0., 0., 0.,
            3.098e-06, 1.859e-05, 5.008e-05, 5.679e-05,
            1.244e-04, 4.486e-04, 2.253e-03, 3.232e-03,
            6.710e-03, 2.290e-02, 5.411e-02, 1.100e-01,
            1.223e-01, 1.281e-01, 1.446e-01, 1.506e-01,
            1.437e-01, 9.266e-02, 1.820e-02, 2.065e-06,
        ]),
        'node_types': np.array([0.7223, 0.1366, 0.1637, 0.1035,
                                0.1422, 0.0054, 0.0015, 0.0]),
        'edge_types': np.array([0.8974, 0.0473, 0.0627, 0.0004, 0.0486]),
    }

    def __init__(self, root: Optional[str] = None, split: str = 'train',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 force_reload: bool = False):
        assert split in ('train', 'val', 'test'), f"Unknown split: {split}"
        self.name = 'moses_gen'
        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        split_idx = {'train': 0, 'val': 1, 'test': 2}[split]
        self.data, self.slices = self.load_data(self.processed_paths[split_idx])

    @property
    def raw_file_names(self) -> List[str]:
        return ['train_moses.csv', 'val_moses.csv', 'test_moses.csv']

    @property
    def processed_file_names(self) -> List[str]:
        return [
            tlx.BACKEND + '_train.pt',
            tlx.BACKEND + '_val.pt',
            tlx.BACKEND + '_test.pt',
        ]

    def download(self):
        train_path = download_url(self.train_url, self.raw_dir)
        os.rename(train_path, osp.join(self.raw_dir, 'train_moses.csv'))

        val_path = download_url(self.val_url, self.raw_dir)
        os.rename(val_path, osp.join(self.raw_dir, 'val_moses.csv'))

        test_path = download_url(self.test_url, self.raw_dir)
        os.rename(test_path, osp.join(self.raw_dir, 'test_moses.csv'))

    def process(self):
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
        except ImportError:
            raise ImportError(
                "MOSESDataset requires RDKit. Install via: pip install rdkit")

        RDLogger.DisableLog('rdApp.*')
        import pandas as pd

        atom_encoder = self.ATOM_ENCODER
        bond_types = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3,
                      BT.AROMATIC: 4}
        num_bond_types = self.NUM_EDGE_TYPES
        num_atom_types = self.NUM_ATOM_TYPES

        split_files = {
            'train': 'train_moses.csv',
            'val': 'val_moses.csv',
            'test': 'test_moses.csv',
        }

        for split_name, split_idx in [('train', 0), ('val', 1), ('test', 2)]:
            csv_path = osp.join(self.raw_dir, split_files[split_name])
            smiles_list = pd.read_csv(csv_path)['SMILES'].values

            data_list = []
            for smile in smiles_list:
                mol = Chem.MolFromSmiles(smile)
                if mol is None:
                    continue

                n_atoms = mol.GetNumAtoms()
                if n_atoms == 0:
                    continue

                # Node features: one-hot atom type
                type_idx = []
                valid = True
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol()
                    if symbol not in atom_encoder:
                        valid = False
                        break
                    type_idx.append(atom_encoder[symbol])
                if not valid:
                    continue

                x = np.zeros((n_atoms, num_atom_types), dtype=np.float32)
                for j, t in enumerate(type_idx):
                    x[j, t] = 1.0

                # Edge index and attributes
                rows, cols, edge_feats = [], [], []
                for bond in mol.GetBonds():
                    start = bond.GetBeginAtomIdx()
                    end = bond.GetEndAtomIdx()
                    bt = bond.GetBondType()
                    if bt not in bond_types:
                        continue
                    bond_idx = bond_types[bt]
                    for s, d in [(start, end), (end, start)]:
                        rows.append(s)
                        cols.append(d)
                        feat = np.zeros(num_bond_types, dtype=np.float32)
                        feat[bond_idx] = 1.0
                        edge_feats.append(feat)

                if len(rows) > 0:
                    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
                    edge_attr = np.stack(edge_feats, axis=0)
                else:
                    continue  # skip molecules with no bonds

                y = np.zeros((1, 0), dtype=np.float32)

                data = Graph(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    n_nodes=np.array([n_atoms], dtype=np.int64),
                    to_tensor=True,
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            if len(data_list) == 0:
                data_list = [Graph(
                    x=np.zeros((1, num_atom_types), dtype=np.float32),
                    edge_index=np.zeros((2, 0), dtype=np.int64),
                    edge_attr=np.zeros((0, num_bond_types), dtype=np.float32),
                    y=np.zeros((1, 0), dtype=np.float32),
                    to_tensor=True,
                )]

            collated_data, slices = self.collate(data_list)
            self.save_data(
                (collated_data, slices),
                self.processed_paths[split_idx],
            )

    def get_stats(self):
        """Return pre-computed dataset statistics for DeFoG training."""
        return self.STATS.copy()

    def __repr__(self) -> str:
        return f'MOSESDataset({len(self)})'
