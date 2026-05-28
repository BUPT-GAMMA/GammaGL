"""Direct tensor construction for GuacaMol, bypassing Graph/collate."""
import os
import sys
sys.path.insert(0, '/data/zhaxi/GammaGL')

import numpy as np
from multiprocessing import Pool
import torch

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
    return x, edge_index, edge_attr, n_atoms


def _build_tensors(results):
    xs, edge_indices, edge_attrs, n_nodes_list = [], [], [], []
    for r in results:
        if r is None:
            continue
        x, ei, ea, n = r
        xs.append(x)
        edge_indices.append(ei)
        edge_attrs.append(ea)
        n_nodes_list.append(n)

    # Build slices
    x_slices = [0]
    ei_slices = [0]
    ea_slices = [0]
    for i in range(len(xs)):
        x_slices.append(x_slices[-1] + xs[i].shape[0])
        ei_slices.append(ei_slices[-1] + edge_indices[i].shape[1])
        ea_slices.append(ea_slices[-1] + edge_attrs[i].shape[0])

    x_cat = np.concatenate(xs, axis=0) if xs else np.zeros((0, NUM_ATOM_TYPES), dtype=np.float32)
    ei_cat = np.concatenate(edge_indices, axis=1) if edge_indices else np.zeros((2, 0), dtype=np.int64)
    ea_cat = np.concatenate(edge_attrs, axis=0) if edge_attrs else np.zeros((0, NUM_EDGE_TYPES), dtype=np.float32)

    # Increment edge_index
    node_offset = 0
    ei_parts = []
    for ei in edge_indices:
        ei_parts.append(ei + node_offset)
        node_offset += xs[len(ei_parts) - 1].shape[0]
    if ei_parts:
        ei_cat = np.concatenate(ei_parts, axis=1)
    else:
        ei_cat = np.zeros((2, 0), dtype=np.int64)

    y = np.zeros((len(xs), 0), dtype=np.float32)
    n_nodes = np.array(n_nodes_list, dtype=np.int64)

    # Create a mock Graph-like object with stores
    from gammagl.data import Graph
    g = Graph(
        x=torch.from_numpy(x_cat),
        edge_index=torch.from_numpy(ei_cat),
        edge_attr=torch.from_numpy(ea_cat),
        y=torch.from_numpy(y),
        n_nodes=torch.from_numpy(n_nodes),
    )

    slices = {
        'x': torch.tensor(x_slices, dtype=torch.int64),
        'edge_index': torch.tensor(ei_slices, dtype=torch.int64),
        'edge_attr': torch.tensor(ea_slices, dtype=torch.int64),
        'y': torch.arange(len(xs) + 1, dtype=torch.int64),
        'n_nodes': torch.arange(len(xs) + 1, dtype=torch.int64),
    }

    return g, slices


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
            results.extend(r)
            print(f"  {min(i+batch, len(smiles))}/{len(smiles)}")

        valid = [r for r in results if r is not None]
        print(f"  Valid: {len(valid)}")

        if not valid:
            raise RuntimeError(f"No valid graphs for {split_name}")

        print("  Building tensors...")
        g, slices = _build_tensors(valid)
        print(f"  x: {g.x.shape}, edges: {g.edge_index.shape[1]}")

        torch.save((g, slices), out_path)
        print(f"  Saved: {out_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
