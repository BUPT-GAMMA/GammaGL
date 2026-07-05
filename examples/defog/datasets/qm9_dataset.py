import os
import os.path as osp
import numpy as np
import tensorlayerx as tlx
from typing import Callable, List, Optional

from gammagl.data import (
    Graph,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class QM9Gen(InMemoryDataset):
    r"""The QM9 dataset for graph generation, following the DeFoG setup.

    From the `"MoleculeNet" <https://arxiv.org/abs/1703.00564>`_ benchmark,
    containing ~134k small organic molecules with up to 9 heavy atoms.
    This version is tailored for **graph generation** (not property prediction):
    node features are one-hot atom types, edge features are one-hot bond types,
    and graph labels are empty.

    Requires **RDKit** (``pip install rdkit``).

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset should be saved.
    split : str
        Which split to load: ``'train'``, ``'val'``, or ``'test'``.
    remove_h : bool
        If ``True``, remove hydrogen atoms. Default ``True`` (paper default).
    aromatic : bool
        If ``True``, include aromatic bond type. Default ``True``.
    transform : callable, optional
        A transform applied to each :class:`Graph` on access.
    pre_transform : callable, optional
        A transform applied during processing before saving.
    pre_filter : callable, optional
        A filter applied during processing.
    force_reload : bool
        Whether to re-process the dataset.
    """

    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'

    # Hard-coded statistics from the DeFoG paper (remove_h=True, aromatic=True)
    STATS_REMOVE_H = {
        'n_nodes': np.array([0, 2.2930e-05, 3.8217e-05, 6.8791e-05, 2.3695e-04,
                             9.7072e-04, 0.0046472, 0.023985, 0.13666, 0.83337]),
        'node_types': np.array([0.7230, 0.1151, 0.1593, 0.0026]),
        'edge_types': np.array([0.7261, 0.2384, 0.0274, 0.0081, 0.0]),
        'valency_distribution': np.pad(
            np.array([2.6071e-06, 0.1630, 0.3520, 0.3200, 0.16313, 0.00073],
                     dtype=np.float32),
            (0, 3 * 9 - 2 - 6),
        ),
        'atom_names': {0: 'C', 1: 'N', 2: 'O', 3: 'F'},
        'valencies': [4, 3, 2, 1],
        'atom_weights': {0: 12, 1: 14, 2: 16, 3: 19},
        'max_weight': 150.0,
        'max_n_nodes': 9,
        'num_node_types': 4,
        'num_edge_types': 5,
    }

    STATS_WITH_H = {
        'n_nodes': np.array([
            0, 0, 0, 1.59e-05, 3.14e-05, 4.41e-05,
            1.13e-04, 2.17e-04, 5.21e-04, 1.38e-03,
            3.43e-03, 8.37e-03, 2.05e-02, 4.62e-02,
            8.80e-02, 1.37e-01, 1.75e-01, 1.78e-01,
            1.51e-01, 1.08e-01, 5.69e-02, 2.24e-02,
            3.76e-03, 1.46e-04, 3.14e-05, 0, 0, 0, 0, 0
        ]),
        'node_types': np.array([0.5122, 0.3526, 0.0562, 0.0777, 0.0013]),
        'edge_types': np.array([0.88162, 0.11062, 5.9875e-03,
                                1.7758e-03, 0.0]),
        'valency_distribution': np.pad(
            np.array([0.0, 0.5136, 0.0840, 0.0554, 0.3456, 0.0012],
                     dtype=np.float32),
            (0, 3 * 29 - 2 - 6),
        ),
        'atom_names': {0: 'H', 1: 'C', 2: 'N', 3: 'O', 4: 'F'},
        'valencies': [1, 4, 3, 2, 1],
        'atom_weights': {0: 1, 1: 12, 2: 14, 3: 16, 4: 19},
        'max_weight': 390.0,
        'max_n_nodes': 29,
        'num_node_types': 5,
        'num_edge_types': 5,
    }

    # Property column indices in gdb9.sdf.csv (0-indexed, after 'mol_id')


    def __init__(self, root: Optional[str] = None, split: str = 'train',
                 remove_h: bool = True, aromatic: bool = True,
                 conditional: bool = False, target: str = 'mu',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 force_reload: bool = False):
        assert split in ('train', 'val', 'test'), f"Unknown split: {split}"
        assert target in ('mu', 'homo', 'both'), f"Unknown target: {target}"
        self.name = 'qm9_gen'
        self.split = split
        self.remove_h = remove_h
        self.aromatic = aromatic
        self.conditional = conditional
        self.target = target

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        split_idx = {'train': 0, 'val': 1, 'test': 2}[split]
        self.data, self.slices = self.load_data(self.processed_paths[split_idx])

    @property
    def processed_dir(self) -> str:
        h_tag = 'no_h' if self.remove_h else 'with_h'
        if self.conditional:
            return osp.join(self.root_dir, f'processed_{h_tag}_cond_{self.target}')
        return osp.join(self.root_dir, f'processed_{h_tag}')

    @property
    def raw_file_names(self) -> List[str]:
        return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def processed_file_names(self) -> List[str]:
        return [
            tlx.BACKEND + '_train.pt',
            tlx.BACKEND + '_val.pt',
            tlx.BACKEND + '_test.pt',
        ]

    def download(self):
        # Download and extract QM9 zip
        file_path = download_url(self.raw_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        if os.path.exists(file_path):
            os.unlink(file_path)
        # Download uncharacterized molecules list
        download_url(self.raw_url2, self.raw_dir,
                     filename='uncharacterized.txt')

    def process(self):
        try:
            from rdkit import Chem
            from rdkit.Chem.rdchem import BondType as BT
        except ImportError:
            raise ImportError(
                "QM9Gen requires RDKit. Install it via: "
                "pip install rdkit  or  conda install -c conda-forge rdkit"
            )

        import pandas as pd

        # Read skip list
        skip_path = osp.join(self.raw_dir, 'uncharacterized.txt')
        with open(skip_path, 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]
        skip = set(skip)

        # Atom and bond type mappings
        full_atom_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        if self.remove_h:
            num_atom_types = 4
        else:
            num_atom_types = 5

        if self.aromatic:
            bond_types = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3,
                          BT.AROMATIC: 4}
            num_bond_types = 5  # 0=no-bond, 1-4=bond types
        else:
            bond_types = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3}
            num_bond_types = 4

        # Read target CSV for conditional labels
        target_path = osp.join(self.raw_dir, 'gdb9.sdf.csv')
        target_df = pd.read_csv(target_path)

        # Read SDF
        sdf_path = osp.join(self.raw_dir, 'gdb9.sdf')
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)

        # Process molecules
        data_list = []

        for i, mol in enumerate(suppl):
            if i in skip or mol is None:
                continue

            atoms = mol.GetAtoms()
            n_atoms = len(atoms)
            if n_atoms == 0:
                continue

            # Node features: keep original atom ordering, then optionally
            # drop hydrogens by heavy-atom subgraph filtering to match DeFoG.
            full_type_idx = []
            keep_mask = []
            for atom in atoms:
                symbol = atom.GetSymbol()
                if symbol not in full_atom_types:
                    break
                full_type_idx.append(full_atom_types[symbol])
                keep_mask.append(symbol != 'H')
            else:
                if self.remove_h:
                    kept_types = [t - 1 for t, keep in zip(full_type_idx, keep_mask) if keep]
                else:
                    kept_types = full_type_idx

                n_kept_atoms = len(kept_types)
                if n_kept_atoms == 0:
                    continue

                x = np.zeros((n_kept_atoms, num_atom_types), dtype=np.float32)
                for j, t in enumerate(kept_types):
                    x[j, t] = 1.0

                if self.remove_h:
                    new_index = {}
                    next_idx = 0
                    for old_idx, keep in enumerate(keep_mask):
                        if keep:
                            new_index[old_idx] = next_idx
                            next_idx += 1
                else:
                    new_index = {idx: idx for idx in range(n_atoms)}

                rows, cols, edge_feats = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if self.remove_h and (not keep_mask[start] or not keep_mask[end]):
                        continue
                    bt = bond.GetBondType()
                    if bt not in bond_types:
                        continue
                    bond_idx = bond_types[bt]
                    mapped_start = new_index[start]
                    mapped_end = new_index[end]
                    for s, d in [(mapped_start, mapped_end), (mapped_end, mapped_start)]:
                        rows.append(s)
                        cols.append(d)
                        feat = np.zeros(num_bond_types, dtype=np.float32)
                        feat[bond_idx] = 1.0
                        edge_feats.append(feat)

                if len(rows) > 0:
                    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
                    edge_attr = np.stack(edge_feats, axis=0)
                    # Sort edges by (source, destination) to match DeFoG
                    perm = (edge_index[0] * n_kept_atoms + edge_index[1]).argsort()
                    edge_index = edge_index[:, perm]
                    edge_attr = edge_attr[perm]
                else:
                    edge_index = np.zeros((2, 0), dtype=np.int64)
                    edge_attr = np.zeros((0, num_bond_types), dtype=np.float32)

                n_atoms = n_kept_atoms

                # Conditional labels or empty y
                if self.conditional:
                    if self.target == 'mu':
                        y_val = np.array([[target_df.iloc[i]['mu']]],
                                         dtype=np.float32)
                    elif self.target == 'homo':
                        y_val = np.array([[target_df.iloc[i]['homo']]],
                                         dtype=np.float32)
                    else:  # both
                        y_val = np.array([[target_df.iloc[i]['mu'],
                                           target_df.iloc[i]['homo']]],
                                         dtype=np.float32)
                else:
                    y_val = np.zeros((1, 0), dtype=np.float32)

                data = Graph(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y_val,
                    n_nodes=np.array([n_atoms], dtype=np.int64),
                    to_tensor=True,
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                continue  # skip the else clause of the for loop
            # If an atom was not recognized, skip this molecule

        # Perform dynamic random split (seed=42) after filtering dirty data
        n_clean_samples = len(data_list)
        n_train = min(100000, n_clean_samples)
        n_test = min(int(0.1 * n_clean_samples), n_clean_samples - n_train)
        n_val = max(0, n_clean_samples - n_train - n_test)

        rng = np.random.RandomState(42)
        indices = rng.permutation(n_clean_samples)

        train_data = [data_list[i] for i in indices[:n_train]]
        val_data = [data_list[i] for i in indices[n_train:n_train + n_val]]
        test_data = [data_list[i] for i in indices[n_train + n_val:]]

        split_data = {'train': train_data, 'val': val_data, 'test': test_data}

        # Save each split
        for split_name, split_idx in [('train', 0), ('val', 1), ('test', 2)]:
            split_list = split_data[split_name]
            if len(split_list) == 0:
                split_list = [Graph(
                    x=np.zeros((1, num_atom_types), dtype=np.float32),
                    edge_index=np.zeros((2, 0), dtype=np.int64),
                    edge_attr=np.zeros((0, num_bond_types), dtype=np.float32),
                    y=np.zeros((1, 0), dtype=np.float32),
                    to_tensor=True,
                )]
            collated_data, slices = self.collate(split_list)
            self.save_data(
                (collated_data, slices),
                self.processed_paths[split_idx],
            )

    def get_stats(self):
        """Return pre-computed dataset statistics for DeFoG training."""
        if self.remove_h:
            return self.STATS_REMOVE_H.copy()
        else:
            return self.STATS_WITH_H.copy()

    def __repr__(self) -> str:
        h_tag = 'no_h' if self.remove_h else 'with_h'
        return f'QM9Gen({len(self)}, {h_tag})'
