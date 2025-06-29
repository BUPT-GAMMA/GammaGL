# Regularizing GNNs via Consistency-Diversity Graph Augmentations (NASA)

This example implements the model from the paper: [Regularizing Graph Neural Networks via Consistency-Diversity Graph Augmentations](https://arxiv.org/abs/2110.07627) (AAAI 2022).

The implementation includes:
- `NR_Augmentor`: A graph transformation that implements the "Neighbor Replacement" (NR) augmentation strategy.
- `NASA_GCN`: A GCN-based model that incorporates the neighbor-constrained regularization loss (L_CR).

## How to Run

You can run the training script from the root directory of the GammaGL repository:

```bash
# Run on Cora dataset
python examples/nasa/nasa_gcn_trainer.py --dataset Cora

# Run on Citeseer dataset
python examples/nasa/nasa_gcn_trainer.py --dataset Citeseer

# Run on PubMed dataset
python examples/nasa/nasa_gcn_trainer.py --dataset PubMed