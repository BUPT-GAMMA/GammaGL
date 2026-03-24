Node Identifiers: Compact, Discrete Representations for Efficient Graph Learning
====================================

This example reproduces the two-stage NodeID pipeline in GammaGL.

- Paper link: [https://arxiv.org/pdf/2405.16435](https://arxiv.org/pdf/2405.16435)
- Author's code repo: [https://github.com/LUOyk1999/NodeID](https://github.com/LUOyk1999/NodeID)

- Stage 1 (`nodeid_trainer.py`): GNN + VQ to generate semantic IDs.
- Stage 2 (`nodeid_id_mlp.py`): ID-MLP on exported semantic IDs.

How to run
----------

```bash
cd examples/nodeid
TL_BACKEND=torch python nodeid_trainer.py --dataset cora --method gcn --n_epoch 1000 --gpu 0
```

Two-stage pipeline
------------------

Run Stage 1 and export semantic IDs:

```bash
TL_BACKEND=torch python nodeid_trainer.py --dataset cora --method gcn --n_epoch 1000 --gpu 0 \
  --rand_split --seed 123 --train_prop 0.6 --valid_prop 0.2 \
  --save_semantic_id --semantic_id_path semantic_ID_cora.npz
```

Run Stage 2 ID-MLP:

```bash
TL_BACKEND=torch python nodeid_id_mlp.py --dataset cora --gpu 0 \
  --rand_split --seed 123 --train_prop 0.6 --valid_prop 0.2 \
  --semantic_id_path semantic_ID_cora.npz \
  --num_layers 2 --hidden_channels 256 --dropout 0.0 \
  --lr 0.001 --l2_coef 5e-4 --n_epoch 1000
```

Results
-------

All results below use:

- `--rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123`
- `1000` training epochs

| Dataset | Method | Original (Highest Valid / Highest Test, %) | Reproduction (Best Valid / Best Test, %) |
| :-----: | :----: | :----------------------------------------: | :---------------------------------------: |
| Cora | GCN | 88.72 / 88.21 | 88.13 ± 0.40 / 87.55 ± 0.50 |
| Citeseer | GCN | 74.44 / 76.88 | 75.31 ± 0.89 / 75.71 ± 1.54 |
| Cora | ID-MLP (from GCN semantic ID) | 87.06 / 86.92 | 86.25 ± 0.98 / 84.53 ± 0.49 |
| Citeseer | ID-MLP (from GCN semantic ID) | 71.88 / 74.32 | 72.03 ± 2.01 / 72.73 ± 2.32 |
| Cora | ID-MLP (from GAT semantic ID) | 87.25 / 85.64 | 87.14 ± 0.56 / 86.23 ± 1.12 |
| Citeseer | ID-MLP (from GAT semantic ID) | 73.53 / 73.72 | 70.41 ± 1.22 / 70.57 ± 0.84 |
| Cora | GAT | 88.17 / 88.77 | 88.32 ± 0.15 / 86.92 ± 0.89 |
| Citeseer | GAT | 75.04 / 77.03 | 73.14 ± 0.35 / 72.85 ± 1.33 |

Notes
-----

- `Original` values are from `NodeID-main/SL/Node_Classification` logs.
- `Reproduction` values are shown as `mean ± std` over fixed-split 5 runs (`split_seed=123`, `train_seed=123..127`).
- Raw per-run values and summary are saved in `repro_stats_fixedsplit_5runs.json`.
- Stage-1 semantic IDs are exported from the best-validation checkpoint.
- GCN/GAT and their ID-MLP rows now share the same fixed-split 5-run protocol.
- Current table is based on reruns dated `2026-03-23`.
- Default examples use GPU (`--gpu 0`); change to `--gpu 1..N` for another device.

Reproduction commands (main settings)
-------------------------------------

```bash
# Cora / GCN
TL_BACKEND=torch python nodeid_trainer.py --dataset cora --method gcn --n_epoch 1000 --display_step 50 --gpu 0 \
  --rand_split --seed 123 --train_prop 0.6 --valid_prop 0.2 \
  --lr 0.001 --local_layers 5 --hidden_channels 256 --l2_coef 5e-4 --dropout 0.0 --num_codes 6

# Cora / GAT
TL_BACKEND=torch python nodeid_trainer.py --dataset cora --method gat --n_epoch 1000 --display_step 50 --gpu 0 \
  --rand_split --seed 123 --train_prop 0.6 --valid_prop 0.2 \
  --lr 0.005 --local_layers 4 --hidden_channels 256 --l2_coef 5e-4 --dropout 0.0 --num_codes 8

# Citeseer / GCN
TL_BACKEND=torch python nodeid_trainer.py --dataset citeseer --method gcn --n_epoch 1000 --display_step 50 --gpu 0 \
  --rand_split --seed 123 --train_prop 0.6 --valid_prop 0.2 \
  --lr 0.005 --local_layers 5 --hidden_channels 64 --l2_coef 0.01 --dropout 0.0 --num_codes 16

# Citeseer / GAT
TL_BACKEND=torch python nodeid_trainer.py --dataset citeseer --method gat --n_epoch 1000 --display_step 50 --gpu 0 \
  --rand_split --seed 123 --train_prop 0.6 --valid_prop 0.2 \
  --lr 0.005 --local_layers 4 --hidden_channels 64 --l2_coef 0.01 --dropout 0.5 --num_codes 8
```
