"""
Graph-to-SMILES conversion and molecular evaluation metrics.
Ported from DeFoG src/analysis/rdkit_functions.py for GammaGL (TensorLayerX).

Provides:
- build_molecule / build_molecule_with_partial_charges
- mol2smiles
- BasicMolecularMetrics (validity, uniqueness, novelty, relaxed validity)
- check_stability / compute_molecular_metrics
"""
import re
import numpy as np

try:
    from rdkit import Chem
    use_rdkit = True
except ImportError:
    use_rdkit = False
    import warnings
    warnings.warn("rdkit not found, molecular evaluation will fail")

# ============================================================
# Molecular constants
# ============================================================

allowed_bonds = {
    "H": 1, "C": 4, "N": 3, "O": 2, "F": 1,
    "B": 3, "Al": 3, "Si": 4, "P": [3, 5], "S": 4,
    "Cl": 1, "As": 3, "Br": 1, "I": 1,
    "Hg": [1, 2], "Bi": [3, 5], "Se": [2, 4, 6],
}

bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE if use_rdkit else None,
    Chem.rdchem.BondType.DOUBLE if use_rdkit else None,
    Chem.rdchem.BondType.TRIPLE if use_rdkit else None,
    Chem.rdchem.BondType.AROMATIC if use_rdkit else None,
]

ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
BOND_ORDER_BY_TYPE = {
    0: 0.0,
    1: 1.0,
    2: 2.0,
    3: 3.0,
    4: 1.5,
}


def _bond_order_from_type(bond_type):
    bond_type = int(bond_type)
    return BOND_ORDER_BY_TYPE.get(bond_type, 0.0)


# ============================================================
# Core molecule building
# ============================================================

def mol2smiles(mol):
    """RDKit Mol -> canonical SMILES string (or None)."""
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def build_molecule(atom_types, edge_types, atom_decoder, verbose=False):
    """Build an RDKit RWMol from integer atom types and integer edge types.

    Parameters
    ----------
    atom_types : array-like, shape (n,)
        Integer node type indices.
    edge_types : array-like, shape (n, n)
        Integer edge type matrix (upper triangle used).
    atom_decoder : list of str
        Maps integer index -> element symbol.
    verbose : bool

    Returns
    -------
    rdkit.Chem.RWMol
    """
    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[int(atom)])
        mol.AddAtom(a)

    edge_types = np.triu(edge_types)
    edge_types = np.copy(edge_types)
    edge_types[edge_types >= len(bond_dict)] = 0  # virtual state → no bond
    rows, cols = np.nonzero(edge_types)
    for i, j in zip(rows, cols):
        if i != j:
            etype = int(edge_types[i, j])
            if etype > 0:
                mol.AddBond(int(i), int(j), bond_dict[etype])

    return mol


def build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder,
                                         verbose=False):
    """Build molecule and add formal charges to fix valence errors."""
    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[int(atom)])
        mol.AddAtom(a)

    edge_types = np.triu(edge_types)
    edge_types = np.copy(edge_types)
    edge_types[edge_types >= len(bond_dict)] = 0
    rows, cols = np.nonzero(edge_types)

    for i, j in zip(rows, cols):
        if i != j:
            etype = int(edge_types[i, j])
            if etype > 0:
                mol.AddBond(int(i), int(j), bond_dict[etype])
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                if atomid_valence and len(atomid_valence) >= 2:
                    idx = atomid_valence[0]
                    v = atomid_valence[1]
                    an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                    if an in (7, 8, 16) and (v - ATOM_VALENCY.get(an, 0)) == 1:
                        mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


# ============================================================
# GDSS valence helpers
# ============================================================

def check_valency(mol):
    """Check valence validity. Returns (ok, atomid_valence_or_None)."""
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence


def correct_mol(m):
    """Attempt to fix valence by removing excess bonds."""
    mol = m
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            if not atomid_valence or len(atomid_valence) < 2:
                # If the error is not a valency error with an index, we cannot correct it
                return None, no_correct
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            check_idx = 0
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                type_ = int(b.GetBondType())
                queue.append((b.GetIdx(), type_, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                if type_ == 12:
                    check_idx += 1
            queue.sort(key=lambda tup: tup[1], reverse=True)

            if queue[-1][1] == 12:
                return None, no_correct
            elif len(queue) > 0:
                start = queue[check_idx][2]
                end = queue[check_idx][3]
                t = queue[check_idx][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_dict[t])
    return mol, no_correct


def check_stability(atom_types, edge_types, atom_decoder):
    """Check molecular stability (correct bond counts per atom).

    Returns (molecule_stable, n_stable_bonds, n_atoms).
    """
    n = len(atom_types)
    n_bonds = np.zeros(n, dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            bond_type = int(round(float(edge_types[i, j] + edge_types[j, i]) / 2.0))
            val = abs(_bond_order_from_type(bond_type))
            n_bonds[i] += val
            n_bonds[j] += val

    n_stable = 0
    for atom_type, n_bond in zip(atom_types, n_bonds):
        possible = allowed_bonds.get(atom_decoder[int(atom_type)], 0)
        if isinstance(possible, int):
            is_stable = np.isclose(float(possible), float(n_bond))
        else:
            is_stable = any(np.isclose(float(candidate), float(n_bond))
                            for candidate in possible)
        n_stable += int(is_stable)

    molecule_stable = n_stable == n
    return molecule_stable, n_stable, n


# ============================================================
# BasicMolecularMetrics
# ============================================================

class BasicMolecularMetrics:
    """Compute validity, uniqueness, novelty on generated molecules."""

    def __init__(self, atom_decoder, train_smiles=None, remove_h=True):
        self.atom_decoder = atom_decoder
        self.train_smiles = train_smiles
        self.remove_h = remove_h

    def compute_validity(self, generated):
        """Check which generated graphs yield valid SMILES.

        Parameters
        ----------
        generated : list of (atom_types, edge_types) tuples
            atom_types: ndarray (n,), edge_types: ndarray (n, n)

        Returns
        -------
        valid : list of str
            Valid SMILES (largest component).
        validity : float
            Fraction valid.
        num_components : ndarray
            Connected components per graph.
        all_smiles : list
            All SMILES (including None for invalid).
        """
        valid = []
        num_components = []
        all_smiles = []

        for graph in generated:
            atom_types, edge_types = graph
            mol = build_molecule(atom_types, edge_types, self.atom_decoder)
            smiles = mol2smiles(mol)

            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                num_components.append(len(mol_frags))
            except Exception:
                pass

            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    all_smiles.append(None)
                except Chem.rdchem.KekulizeException:
                    all_smiles.append(None)
            else:
                all_smiles.append(None)

        validity = len(valid) / max(len(generated), 1)
        return valid, validity, np.array(num_components), all_smiles

    def compute_uniqueness(self, valid):
        """Deduplicate valid SMILES."""
        unique = list(set(valid))
        uniqueness = len(unique) / max(len(valid), 1)
        return unique, uniqueness

    def compute_novelty(self, unique):
        """Check how many unique SMILES are not in training set."""
        if self.train_smiles is None:
            print("Dataset smiles is None, novelty computation skipped")
            return [], 1.0
        novel = [s for s in unique if s not in self.train_smiles]
        novelty = len(novel) / max(len(unique), 1)
        return novel, novelty

    def compute_relaxed_validity(self, generated):
        """Validity with partial charge correction."""
        valid = []
        for graph in generated:
            atom_types, edge_types = graph
            mol = build_molecule_with_partial_charges(
                atom_types, edge_types, self.atom_decoder)
            smiles = mol2smiles(mol)
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol,
                                      key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                except (Chem.rdchem.AtomValenceException,
                        Chem.rdchem.KekulizeException):
                    pass
        relaxed_validity = len(valid) / max(len(generated), 1)
        return valid, relaxed_validity

    def evaluate(self, generated):
        """Run all metrics.

        Parameters
        ----------
        generated : list of (atom_types, edge_types)

        Returns
        -------
        metrics : list [validity, relaxed_validity, uniqueness, novelty]
        unique : list of str
        nc_dict : dict (nc_min, nc_max, nc_mu)
        all_smiles : list
        """
        valid, validity, num_components, all_smiles = self.compute_validity(generated)

        nc_mu = float(num_components.mean()) if len(num_components) > 0 else 0
        nc_min = float(num_components.min()) if len(num_components) > 0 else 0
        nc_max = float(num_components.max()) if len(num_components) > 0 else 0

        print(f"  Validity: {validity * 100:.2f}%  ({len(generated)} molecules)")
        print(f"  Connected components: min={nc_min:.2f} mean={nc_mu:.2f} max={nc_max:.2f}")

        relaxed_valid, relaxed_validity = self.compute_relaxed_validity(generated)
        print(f"  Relaxed validity: {relaxed_validity * 100:.2f}%")

        if relaxed_validity > 0:
            unique, uniqueness = self.compute_uniqueness(relaxed_valid)
            print(f"  Uniqueness: {uniqueness * 100:.2f}%  ({len(relaxed_valid)} valid)")

            if self.train_smiles is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"  Novelty: {novelty * 100:.2f}%  ({len(unique)} unique)")
            else:
                novelty = -1.0
        else:
            novelty = -1.0
            uniqueness = 0.0
            unique = []

        return (
            [validity, relaxed_validity, uniqueness, novelty],
            unique,
            dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu),
            all_smiles,
        )


# ============================================================
# Top-level molecular metrics function
# ============================================================

def compute_molecular_metrics(molecule_list, train_smiles, atom_decoder,
                               remove_h=True):
    """Compute all molecular metrics on generated graphs.

    Parameters
    ----------
    molecule_list : list of (atom_types, edge_types)
        Generated graphs.
    train_smiles : list of str or None
        Training set SMILES for novelty computation.
    atom_decoder : list of str
        Element symbol for each node type index.
    remove_h : bool
        Whether hydrogen was removed from the dataset.

    Returns
    -------
    stability_dict : dict
        mol_stable, atm_stable (or -1 if remove_h).
    rdkit_metrics : tuple from BasicMolecularMetrics.evaluate().
    all_smiles : list
    summary : dict
    """
    if not remove_h:
        print("Analyzing molecule stability...")
        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0

        for mol_data in molecule_list:
            atom_types, edge_types = mol_data
            s, nb, na = check_stability(atom_types, edge_types, atom_decoder)
            molecule_stable += int(s)
            nr_stable_bonds += int(nb)
            n_atoms += int(na)

        stability_dict = {
            "mol_stable": molecule_stable / max(len(molecule_list), 1),
            "atm_stable": nr_stable_bonds / max(n_atoms, 1),
        }
    else:
        stability_dict = {"mol_stable": -1, "atm_stable": -1}

    metrics = BasicMolecularMetrics(atom_decoder, train_smiles, remove_h)
    rdkit_metrics = metrics.evaluate(molecule_list)
    metric_values, unique_smiles, nc, all_smiles = rdkit_metrics

    summary = {
        "Validity": metric_values[0],
        "Relaxed Validity": metric_values[1],
        "Uniqueness": metric_values[2],
        "Novelty": metric_values[3],
        "nc_max": nc["nc_max"],
        "nc_mu": nc["nc_mu"],
    }

    return stability_dict, rdkit_metrics, all_smiles, summary


# ============================================================
# FCD (Frechet ChemNet Distance)
# ============================================================

def _fcd_worker(generated_smiles, reference_smiles, queue):
    """Worker for FCD computation in a spawn subprocess."""
    try:
        from fcd import get_fcd
        fcd_score = get_fcd(generated_smiles, reference_smiles)
        queue.put(('ok', fcd_score))
    except Exception as e:
        queue.put(('error', str(e)))


def compute_fcd(generated_smiles, reference_smiles):
    """Compute Frechet ChemNet Distance between generated and reference SMILES.

    Parameters
    ----------
    generated_smiles : list of str
        Generated canonical SMILES (None entries will be filtered).
    reference_smiles : list of str
        Reference (test/val) canonical SMILES.

    Returns
    -------
    float
        FCD score, or -1 if computation fails.
    """
    try:
        import fcd  # noqa: F401
    except ImportError:
        print("  FCD: fcd package not installed. Install with: pip install fcd")
        return -1.0

    import multiprocessing as mp
    import time

    print("  Starting FCD computation...")
    start = time.time()

    generated_smiles = [s for s in generated_smiles if s is not None]

    # Use spawn context to avoid CUDA re-initialization issues
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    p = ctx.Process(target=_fcd_worker, args=(generated_smiles, reference_smiles, queue))
    p.start()
    p.join(timeout=300)

    if p.is_alive():
        p.terminate()
        print("  FCD computation timed out. Setting FCD to -1.")
        fcd_score = -1.0
    else:
        try:
            status, result = queue.get(timeout=5)
            if status == 'ok':
                fcd_score = result
            else:
                print(f"  Error in FCD computation: {result}. Setting FCD to -1.")
                fcd_score = -1.0
        except Exception as e:
            print(f"  Error retrieving FCD result: {e}. Setting FCD to -1.")
            fcd_score = -1.0

    elapsed = time.time() - start
    print(f"  FCD: {fcd_score:.4f} (took {elapsed:.1f}s)")
    return fcd_score


# ============================================================
# Distribution MAE metrics (node/edge/valency/n distribution)
# ============================================================

def compute_distribution_metrics(generated_graphs, dataset_infos, dataset_name):
    """Compute distribution MAE metrics: node count, atom type, bond type, valency.

    Parameters
    ----------
    generated_graphs : list of (X_int, E_int) tuples
        Generated graphs.
    dataset_infos : dict
        Dataset statistics with keys 'max_n_nodes', 'node_types', 'edge_types'.
    dataset_name : str
        Dataset name for atom decoder lookup.

    Returns
    -------
    dict
        Distribution MAE metrics.
    """
    max_n = dataset_infos['max_n_nodes']
    num_node_types = len(dataset_infos['node_types'])
    num_edge_types = len(dataset_infos['edge_types'])

    # Reference distributions (normalized)
    ref_node_types = dataset_infos['node_types'].copy()
    ref_node_types = ref_node_types / (ref_node_types.sum() + 1e-8)

    ref_edge_types = dataset_infos['edge_types'].copy()
    ref_edge_types = ref_edge_types / (ref_edge_types.sum() + 1e-8)

    # Node count distribution
    ref_node_dist = dataset_infos['node_dist'].copy()

    # Accumulate generated distributions
    gen_n_dist = np.zeros_like(ref_node_dist)
    gen_node_dist = np.zeros(num_node_types, dtype=np.float32)
    gen_edge_dist = np.zeros(num_edge_types, dtype=np.float32)
    gen_valency_dist = np.zeros(max(3 * max_n - 2, 1), dtype=np.float32)
    bond_orders = np.array([0.0, 1.0, 2.0, 3.0, 1.5], dtype=np.float32)

    for (x, e) in generated_graphs:
        n = len(x)
        if n < len(gen_n_dist):
            gen_n_dist[n] += 1

        # Atom type distribution
        for atom_type in x:
            at = int(atom_type)
            if 0 <= at < num_node_types:
                gen_node_dist[at] += 1

        # Bond type distribution (upper triangle only)
        for i in range(n):
            for j in range(i + 1, n):
                bt = int(e[i, j])
                if 0 <= bt < num_edge_types:
                    gen_edge_dist[bt] += 1

        # Valency distribution
        for i in range(n):
            val = 0.0
            for j in range(n):
                bt = int(e[i, j])
                if 0 <= bt < len(bond_orders):
                    val += bond_orders[bt]
            val_idx = int(val)
            if 0 <= val_idx < len(gen_valency_dist):
                gen_valency_dist[val_idx] += 1

    # Normalize
    gen_n_dist_norm = gen_n_dist / (gen_n_dist.sum() + 1e-8)
    ref_n_dist_norm = ref_node_dist / (ref_node_dist.sum() + 1e-8)

    gen_node_norm = gen_node_dist / (gen_node_dist.sum() + 1e-8)
    gen_edge_norm = gen_edge_dist / (gen_edge_dist.sum() + 1e-8)
    gen_val_norm = gen_valency_dist / (gen_valency_dist.sum() + 1e-8)

    ref_val_norm = np.zeros_like(gen_valency_dist)
    ref_valency = dataset_infos.get('valency_distribution')
    if ref_valency is not None:
        ref_valency = np.asarray(ref_valency, dtype=np.float32).reshape(-1)
        ref_len = min(len(ref_valency), len(ref_val_norm))
        ref_val_norm[:ref_len] = ref_valency[:ref_len]
        ref_val_norm = ref_val_norm / (ref_val_norm.sum() + 1e-8)
    else:
        # Fallback for datasets without a stored reference valency histogram.
        ref_val_norm[0] = ref_node_types[0]
        ref_val_norm = ref_val_norm / (ref_val_norm.sum() + 1e-8)

    metrics = {}

    # Node count MAE
    min_len = min(len(gen_n_dist_norm), len(ref_n_dist_norm))
    metrics['n_dist_mae'] = float(np.abs(
        gen_n_dist_norm[:min_len] - ref_n_dist_norm[:min_len]).mean())

    # Atom type MAE
    min_len = min(len(gen_node_norm), len(ref_node_types))
    metrics['node_dist_mae'] = float(np.abs(
        gen_node_norm[:min_len] - ref_node_types[:min_len]).mean())

    # Bond type MAE
    min_len = min(len(gen_edge_norm), len(ref_edge_types))
    metrics['edge_dist_mae'] = float(np.abs(
        gen_edge_norm[:min_len] - ref_edge_types[:min_len]).mean())

    # Valency MAE
    metrics['valency_dist_mae'] = float(np.abs(
        gen_val_norm - ref_val_norm).mean())

    return metrics
