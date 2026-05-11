"""
Example: Molecular property prediction with MoleculeGNN on GammaGL.

This script demonstrates how to use the newly added MoleculeGNN components
for a typical molecular graph classification / regression task.  It uses
synthetic data to illustrate the full pipeline: data loading, model
construction, training loop, and evaluation.

Usage
-----
    python examples/molecule_gnn/molecule_gnn_trainer.py

    # or override defaults
    python examples/molecule_gnn/molecule_gnn_trainer.py --epochs 50 --lr 1e-4
"""

import argparse
import os
import sys

import tensorlayerx as tlx
import numpy as np

# Ensure the library root is on the Python path when running the script directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from gammagl.models.molecule_gnn import MoleculeGNN


# ---------------------------------------------------------------------------
# Synthetic molecular graph dataset
# ---------------------------------------------------------------------------

def build_synthetic_dataset(num_graphs=200, num_atom_features=9, num_bond_types=4):
    """Build a tiny synthetic dataset of molecular graphs for demonstration.

    Each "molecule" is a small random graph with atom features and bond types.
    The label is a deterministic function of the graph (for reproducibility).

    Returns
    -------
    graphs : list[dict]
        Each dict has keys:
            - x         : (num_atoms, num_atom_features)  float32
            - edge_index: (2, num_edges)                  int64
            - edge_type : (num_edges,)                    int64
            - y         : (1,)                            float32
    """
    np.random.seed(42)
    graphs = []
    for _ in range(num_graphs):
        num_atoms = np.random.randint(5, 20)
        num_edges = np.random.randint(num_atoms, num_atoms * 2)

        # Random atom features (simulating one-hot atom type + degree + chirality etc.)
        x = np.random.randn(num_atoms, num_atom_features).astype(np.float32)

        # Random edges (undirected)
        src = np.random.randint(0, num_atoms, size=num_edges)
        dst = np.random.randint(0, num_atoms, size=num_edges)
        # Remove self-loops
        mask = src != dst
        src, dst = src[mask], dst[mask]
        num_edges = src.shape[0]

        edge_index = np.stack([src, dst], axis=0).astype(np.int64)

        # Random bond types
        edge_type = np.random.randint(0, num_bond_types, size=num_edges).astype(np.int64)

        # Simple deterministic label
        y = np.array([x.sum() * 0.01], dtype=np.float32)

        graphs.append({"x": x, "edge_index": edge_index, "edge_type": edge_type, "y": y})

    return graphs


def collate_batch(graphs):
    """Merge a list of graph dicts into a single batched representation."""
    xs, edge_indices, edge_types, ys, batches = [], [], [], [], []
    node_offset = 0
    for i, g in enumerate(graphs):
        xs.append(g["x"])
        edge_indices.append(g["edge_index"] + node_offset)
        edge_types.append(g["edge_type"])
        ys.append(g["y"])
        batches.append(np.full(g["x"].shape[0], i, dtype=np.int64))
        node_offset += g["x"].shape[0]

    return {
        "x": tlx.convert_to_tensor(np.concatenate(xs, axis=0)),
        "edge_index": tlx.convert_to_tensor(np.concatenate(edge_indices, axis=1)),
        "edge_type": tlx.convert_to_tensor(np.concatenate(edge_types, axis=0)),
        "y": tlx.convert_to_tensor(np.concatenate(ys, axis=0)),
        "batch": tlx.convert_to_tensor(np.concatenate(batches, axis=0)),
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, graphs, optimizer, batch_size=32):
    model.train()
    total_loss = 0.0
    num_batches = 0
    indices = np.random.permutation(len(graphs))
    for start in range(0, len(graphs), batch_size):
        batch_graphs = [graphs[i] for i in indices[start:start + batch_size]]
        data = collate_batch(batch_graphs)

        with tlx.Backend() as B:
            pred = model(data["x"], data["edge_index"], data["edge_type"], data["batch"])
            loss = B.reduce_mean((pred - data["y"]) ** 2)

        B.zero_grad(optimizer)
        B.backward(loss, optimizer)
        optimizer.step()

        total_loss += float(loss)
        num_batches += 1

    return total_loss / max(num_batches, 1)


@tlx.no_grad
def evaluate(model, graphs, batch_size=32):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for start in range(0, len(graphs), batch_size):
        batch_graphs = graphs[start:start + batch_size]
        data = collate_batch(batch_graphs)

        pred = model(data["x"], data["edge_index"], data["edge_type"], data["batch"])
        loss = tlx.reduce_mean((pred - data["y"]) ** 2)

        total_loss += float(loss)
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MoleculeGNN trainer")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-graphs", type=int, default=200)
    parser.add_argument("--num-atom-features", type=int, default=9)
    parser.add_argument("--num-bond-types", type=int, default=4)
    args = parser.parse_args()

    # Synthetic data
    all_graphs = build_synthetic_dataset(
        num_graphs=args.num_graphs,
        num_atom_features=args.num_atom_features,
        num_bond_types=args.num_bond_types,
    )
    split = int(0.8 * len(all_graphs))
    train_graphs = all_graphs[:split]
    test_graphs = all_graphs[split:]

    # Model
    model = MoleculeGNN(
        node_in_dim=args.num_atom_features,
        hidden_dim=args.hidden_dim,
        out_dim=1,                       # single regression target
        num_layers=args.num_layers,
        num_bond_types=args.num_bond_types,
        dropout=args.dropout,
        residual=True,
    )

    optimizer = tlx.optimizers.Adam(args.lr)

    print(f"Training MoleculeGNN on {len(train_graphs)} graphs "
          f"(eval on {len(test_graphs)} graphs)")
    print(f"  layers={args.num_layers}, hidden={args.hidden_dim}, "
          f"dropout={args.dropout}, lr={args.lr}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_graphs, optimizer, args.batch_size)
        test_loss = evaluate(model, test_graphs, args.batch_size)
        print(f"Epoch {epoch:03d} | train loss={train_loss:.6f} | test loss={test_loss:.6f}")

    print("Done.")


if __name__ == "__main__":
    main()
