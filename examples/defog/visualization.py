"""
Graph and molecule visualization utilities.
Ported from DeFoG src/analysis/visualization.py for GammaGL.

Key changes from original:
- Removed torch dependency (uses numpy)
- Removed wandb dependency (saves to disk only)
- Removed TLS dataset dependency (simplified)
"""
import os
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit import RDLogger
    use_rdkit = True
except ImportError:
    use_rdkit = False

import networkx as nx
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt


class MolecularVisualization:
    """Visualize generated molecular graphs as 2D depictions."""

    def __init__(self, remove_h, atom_decoder):
        self.remove_h = remove_h
        self.atom_decoder = atom_decoder

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """Convert node types + adjacency matrix to an RDKit Mol.

        Parameters
        ----------
        node_list : array-like (n,)
            Integer node type indices.
        adjacency_matrix : array-like (n, n)
            Integer edge type matrix.

        Returns
        -------
        rdkit.Chem.Mol or None
        """
        if not use_rdkit:
            return None

        bond_type_map = {
            1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE,
            4: Chem.rdchem.BondType.AROMATIC,
        }

        mol = Chem.RWMol()
        node_to_idx = {}
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            a = Chem.Atom(self.atom_decoder[int(node_list[i])])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                if iy <= ix:
                    continue
                bond_type = bond_type_map.get(int(bond))
                if bond_type is not None:
                    mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        try:
            mol = mol.GetMol()
        except Exception:
            mol = None
        return mol

    def visualize(self, path, molecules, num_molecules_to_visualize):
        """Save generated molecules as PNG images.

        Parameters
        ----------
        path : str
            Directory to save images.
        molecules : list of (node_types, adj_matrix) tuples
        num_molecules_to_visualize : int
        """
        if not use_rdkit:
            print("  RDKit not available, skipping molecular visualization")
            return

        if not os.path.exists(path):
            os.makedirs(path)

        num_molecules_to_visualize = min(num_molecules_to_visualize, len(molecules))
        print(f"  Visualizing {num_molecules_to_visualize} molecules...")

        for i in range(num_molecules_to_visualize):
            file_path = os.path.join(path, f"molecule_{i}.png")
            node_types = molecules[i][0]
            adj = molecules[i][1]
            if isinstance(node_types, np.ndarray) is False:
                node_types = np.asarray(node_types)
            if isinstance(adj, np.ndarray) is False:
                adj = np.asarray(adj)
            mol = self.mol_from_graphs(node_types, adj)
            if mol is not None:
                try:
                    Draw.MolToFile(mol, file_path)
                except Exception:
                    pass


class NonMolecularVisualization:
    """Visualize generated synthetic graphs."""

    def __init__(self):
        pass

    def to_networkx(self, node_list, adjacency_matrix):
        """Convert node types + adjacency to networkx graph.

        Parameters
        ----------
        node_list : array-like (n,)
        adjacency_matrix : array-like (n, n)

        Returns
        -------
        nx.Graph
        """
        graph = nx.Graph()
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            graph.add_node(i, number=i, symbol=node_list[i], color_val=node_list[i])

        adj = np.asarray(adjacency_matrix)
        rows, cols = np.where(adj >= 1)
        for s, d in zip(rows.tolist(), cols.tolist()):
            edge_type = adj[s, d]
            graph.add_edge(s, d, color=float(edge_type), weight=3 * edge_type)

        return graph

    def visualize_non_molecule(self, graph, pos, path, iterations=100,
                                node_size=100, largest_component=False, time=None):
        """Draw a single graph to file."""
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]

        if pos is None:
            pos = nx.spring_layout(graph, iterations=iterations)

        # Node colors from eigenvector
        w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).toarray())
        vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
        m = max(np.abs(vmin), vmax)

        plt.figure()
        nx.draw(
            graph, pos, font_size=5, node_size=node_size,
            with_labels=False, node_color=U[:, 1],
            cmap=plt.cm.coolwarm, vmin=-m, vmax=m, edge_color="grey",
        )
        if time is not None:
            plt.text(0.5, 0.05, f"t = {time:.2f}",
                     ha="center", va="center",
                     transform=plt.gcf().transFigure, fontsize=16)
        plt.tight_layout()
        plt.savefig(path)
        plt.close("all")

    def visualize(self, path, graphs, num_graphs_to_visualize):
        """Save generated graphs as PNG images.

        Parameters
        ----------
        path : str
            Directory to save images.
        graphs : list of (node_types, adj_matrix) tuples
        num_graphs_to_visualize : int
        """
        if not os.path.exists(path):
            os.makedirs(path)

        num_graphs_to_visualize = min(num_graphs_to_visualize, len(graphs))
        print(f"  Visualizing {num_graphs_to_visualize} graphs...")

        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, f"graph_{i}.png")
            node_types = graphs[i][0]
            adj = graphs[i][1]
            if not isinstance(node_types, np.ndarray):
                node_types = np.asarray(node_types)
            if not isinstance(adj, np.ndarray):
                adj = np.asarray(adj)
            graph = self.to_networkx(node_types, adj)
            self.visualize_non_molecule(graph=graph, pos=None, path=file_path)
