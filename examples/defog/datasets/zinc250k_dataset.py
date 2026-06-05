import os
import os.path as osp
import json
import numpy as np
import tensorlayerx as tlx
from typing import Callable, List, Optional

from gammagl.data import (
    Graph,
    InMemoryDataset,
    download_url,
)


class ZINC250kGen(InMemoryDataset):
    r"""The ZINC250k dataset for molecular graph generation.

    Contains ~250k drug-like molecules from the ZINC database, processed
    with kekulization (no aromatic bonds).

    From `"Grammar Variational Autoencoder"
    <https://arxiv.org/abs/1703.01925>`_ .

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

    CSV_URL = ('https://raw.githubusercontent.com/harryjo97/GruM/'
               'master/GruM_2D/data/zinc250k.csv')
    IDX_URL = ('https://raw.githubusercontent.com/harryjo97/GruM/'
               'master/GruM_2D/data/valid_idx_zinc250k.json')

    ATOM_DECODER = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    ATOM_ENCODER = {atom: i for i, atom in enumerate(ATOM_DECODER)}
    NUM_ATOM_TYPES = 9
    NUM_EDGE_TYPES = 4  # no-bond=0, single=1, double=2, triple=3 (kekulized)

    STATS = {
        'valencies': [4, 3, 2, 1, 5, 6, 1, 1, 1],
        'atom_weights': {0: 12, 1: 14, 2: 16, 3: 19, 4: 30, 5: 32,
                         6: 35.5, 7: 78, 8: 127},
        'max_weight': 500.0,
        'max_n_nodes': 38,
        'num_node_types': 9,
        'num_edge_types': 4,
        'n_nodes': np.array([
            0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
            0.0000e+00, 1.3359e-05, 2.2265e-05, 5.7889e-05, 2.9835e-04,
            7.9263e-04, 2.9123e-03, 4.6890e-03, 7.1515e-03, 1.1275e-02,
            1.7117e-02, 2.5360e-02, 3.5014e-02, 4.6707e-02, 5.8178e-02,
            7.0829e-02, 8.1472e-02, 7.4922e-02, 8.4384e-02, 9.3099e-02,
            9.1451e-02, 7.7175e-02, 6.3397e-02, 4.0331e-02, 3.1131e-02,
            2.4394e-02, 1.9237e-02, 1.5029e-02, 1.0362e-02, 6.9155e-03,
            4.1190e-03, 1.5942e-03, 5.6108e-04, 8.9060e-06,
        ], dtype=np.float32),
        'node_types': np.array([
            7.3678e-01, 1.2211e-01, 9.9746e-02, 1.3745e-02, 2.4428e-05,
            1.7806e-02, 7.4231e-03, 2.2057e-03, 1.5522e-04,
        ], dtype=np.float32),
        'edge_types': np.array([
            9.0658e-01, 6.9411e-02, 2.3771e-02, 2.3480e-04,
        ], dtype=np.float32),
        'valency_distribution': np.concatenate([
            np.array([
                0.0000e+00, 1.1364e-01, 3.0431e-01, 3.5063e-01,
                2.2655e-01, 2.2697e-05, 4.8356e-03,
            ], dtype=np.float32),
            np.zeros(105, dtype=np.float32),
        ]),
    }

    def __init__(self, root: Optional[str] = None, split: str = 'train',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 force_reload: bool = False):
        assert split in ('train', 'val', 'test'), f"Unknown split: {split}"
        self.name = 'zinc250k_gen'
        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        split_idx = {'train': 0, 'val': 1, 'test': 2}[split]
        self.data, self.slices = self.load_data(self.processed_paths[split_idx])

    @property
    def raw_file_names(self) -> List[str]:
        return ['zinc250k.csv', 'valid_idx_zinc250k.json']

    @property
    def processed_file_names(self) -> List[str]:
        return [
            tlx.BACKEND + '_train.pt',
            tlx.BACKEND + '_val.pt',
            tlx.BACKEND + '_test.pt',
        ]

    def download(self):
        download_url(self.CSV_URL, self.raw_dir)
        download_url(self.IDX_URL, self.raw_dir)

    def process(self):
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
        except ImportError:
            raise ImportError(
                "ZINC250kGen requires RDKit. Install via: pip install rdkit")

        RDLogger.DisableLog('rdApp.*')
        import pandas as pd

        atom_encoder = self.ATOM_ENCODER
        bond_types = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3}
        num_bond_types = self.NUM_EDGE_TYPES
        num_atom_types = self.NUM_ATOM_TYPES

        # Read CSV
        csv_path = osp.join(self.raw_dir, 'zinc250k.csv')
        df = pd.read_csv(csv_path)
        smiles_list = df['smiles'].values

        # Read validation indices
        idx_path = osp.join(self.raw_dir, 'valid_idx_zinc250k.json')
        with open(idx_path, 'r') as f:
            val_indices = set(json.load(f))

        # val and test share the same indices (original DeFoG behavior)
        train_indices = [i for i in range(len(smiles_list))
                         if i not in val_indices]
        val_indices_list = sorted(val_indices)

        split_indices = {
            'train': train_indices,
            'val': val_indices_list,
            'test': val_indices_list,
        }

        for split_name, split_idx in [('train', 0), ('val', 1), ('test', 2)]:
            indices = split_indices[split_name]
            data_list = []

            for i in indices:
                smile = smiles_list[i]
                mol = Chem.MolFromSmiles(smile)
                if mol is None:
                    continue

                # Kekulize to remove aromatic bonds
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except Exception:
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

                # Edge index and attributes (kekulized: no aromatic)
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
                    continue

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
        return f'ZINC250kGen({len(self)})'
