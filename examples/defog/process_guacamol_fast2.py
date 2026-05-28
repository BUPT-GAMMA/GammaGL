"""Fast parallel processing of GuacaMol SMILES."""
import os
import sys
sys.path.insert(0, '/data/zhaxi/GammaGL')

import numpy as np
from multiprocessing import Pool
from gammagl.data import Graph, InMemoryDataset

ATOM_ENCODER = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'B': 4, 'Br': 5, 'Cl': 6,
                'I': 7, 'P': 8, 'S': 9, 'Se': 10, 'Si': 11}
NUM_ATOM_TYPES = 12
NUM_EDGE_TYPES = 5


def _process_one(smile):
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondType as BT
    bond_types = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}

    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None

    type_idx = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ATOM_ENCODER:
            return None
        type_idx.append(ATOM_ENCODER[symbol])

    x = np.zeros((n_atoms, NUM_ATOM_TYPES), dtype=np.float32)
    for j, t in enumerate(type_idx):
        x[j, t] = 1.0

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
            feat = np.zeros(NUM_EDGE_TYPES, dtype=np.float32)
            feat[bond_idx] = 1.0
            edge_feats.append(feat)

    if len(rows) == 0:
        return None

    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_attr = np.stack(edge_feats, axis=0)
    return dict(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=np.zeros((1, 0), dtype=np.float32),
                n_nodes=np.array([n_atoms], dtype=np.int64))


def _save_split(data_dicts, out_path):
    data_list = [Graph(**d, to_tensor=True) for d in data_dicts]
    class _DS(InMemoryDataset):
        def __init__(self): pass
    ds = _DS()
    collated, slices = ds.collate(data_list)
    ds.save_data((collated, slices), out_path)


def main():
    raw_dir = './datasets/guacamol/guacamol_gen/raw'
    processed_dir = './datasets/guacamol/guacamol_gen/processed'
    os.makedirs(processed_dir, exist_ok=True)

    n_workers = min(32, os.cpu_count())
    print(f"Workers: {n_workers}")

    for split_name, smiles_file, out_file in [
        ('train', 'train.smiles', 'torch_train.pt'),
        ('val', 'valid.smiles', 'torch_val.pt'),
        ('test', 'test.smiles', 'torch_test.pt'),
    ]:
        smiles_path = os.path.join(raw_dir, smiles_file)
        out_path = os.path.join(processed_dir, out_file)

        with open(smiles_path) as f:
            smiles = [l.strip() for l in f if l.strip()]
        print(f"\n{split_name}: {len(smiles)} SMILES")

        results = []
        batch = 50000
        for i in range(0, len(smiles), batch):
            chunk = smiles[i:i + batch]
            with Pool(n_workers) as pool:
                r = pool.map(_process_one, chunk)
            results.extend([x for x in r if x is not None])
            print(f"  {min(i+batch, len(smiles))}/{len(smiles)} -> {len(results)} graphs")

        if not results:
            raise RuntimeError(f"No valid graphs for {split_name}")
        print(f"  Saving {len(results)} graphs...")
        _save_split(results, out_path)
        print(f"  Done: {out_path}")

    print("\nAll splits processed!")


if __name__ == '__main__':
    main()
