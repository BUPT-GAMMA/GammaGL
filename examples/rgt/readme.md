# Riemannian Graph Transformer (RGT)

- Paper link: [https://arxiv.org/abs/2309.15819](https://arxiv.org/abs/2309.15819)

- Official pytorch implementation:
  [https://github.com/xyt2022/RGT](https://github.com/xyt2022/RGT)

## Overview

RGT (Riemannian Graph Transformer) is a graph neural network that operates on Riemannian manifolds (hyperbolic and spherical spaces) to capture complex structural patterns in graph data. It combines vector quantization with cross-manifold attention for self-supervised pre-training.

## Dataset Statistics

| Dataset  | #Nodes | #Edges | #Classes | #Features |
|----------|--------|--------|----------|-----------|
| Cora     | 2,708  | 5,429  | 7        | 1,433     |
| Citeseer | 3,327  | 4,732  | 6        | 3,703     |
| PubMed   | 19,717 | 44,338 | 3        | 500       |
| Brazil   | 13,141 | 45,513 | 2        | 1,024     |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid) for more details.

## Results

| Dataset  | Task               | Metric         | Paper (RGT)   | Ours (Reproduced) |
|----------|--------------------|----------------|---------------|-------------------|
| Cora     | Node Classification| Accuracy (%)   | 80.72 ± 0.97  | 75.72 ± 0.60      |
| Cora     | Node Classification| Weighted F1 (%)| 80.79 ± 0.92  | 75.92 ± 0.60      |
| Citeseer | Node Classification| Accuracy (%)   | 66.40 ± 1.53  | 60.96 ± 1.08      |
| Citeseer | Node Classification| Weighted F1 (%)| 67.81 ± 0.94  | 61.66 ± 0.77      |
| Cora     | Link Prediction    | AUC (%)        | 96.82 ± 0.01  | 95.76 ± 0.04      |
| Cora     | Link Prediction    | AP (%)         | 96.26 ± 0.01  | 94.94 ± 0.05      |
| Citeseer | Link Prediction    | AUC (%)        | 99.48 ± 0.00  | 98.78 ± 0.03      |
| Citeseer | Link Prediction    | AP (%)         | 99.39 ± 0.01  | 98.44 ± 0.04      |
| Brazil   | Link Prediction    | AUC (%)        | 84.89 ± 0.11  | 83.06 ± 0.63      |
| Brazil   | Link Prediction    | AP (%)         | 83.61 ± 0.10  | 81.05 ± 0.40      |

## Usage

### Pre-training

```bash
python examples/rgt/train.py \
    --task Pretrain \
    --pretrain_dataset Cora Citeseer \
    --batch_size 64 \
    --embed_dim 32 \
    --n_layers 2 \
    --pretrain_epochs 1 \
    --pretrain_iters 1 \
    --gpu 0
```

> To use larger pretraining datasets (ogbn-arxiv, computers, Physics), ensure OGB is installed and network access to Stanford/GitHub is stable.

### Node Classification

```bash
# Cora
python examples/rgt/train.py \
    --task NC \
    --dataset Cora \
    --gpu 0 \
    --pretrained_model_path Pretrain_ogbn-arxiv_computers_Physics_model \
    --nc_epochs 120

# Citeseer
python examples/rgt/train.py \
    --task NC \
    --dataset Citeseer \
    --gpu 0 \
    --pretrained_model_path Pretrain_ogbn-arxiv_computers_Physics_model \
    --nc_epochs 120
```

> The pretrained model checkpoint (`.pt`) should be placed under `checkpoints/`.

### Link Prediction

```bash
# Cora
python examples/rgt/train.py \
    --task LP \
    --dataset Cora \
    --gpu 0 \
    --pretrained_model_path Pretrain_ogbn-arxiv_computers_Physics_model \
    --lp_epochs 3

# Citeseer
python examples/rgt/train.py \
    --task LP \
    --dataset Citeseer \
    --gpu 0 \
    --pretrained_model_path Pretrain_ogbn-arxiv_computers_Physics_model \
    --lp_epochs 3

# Brazil
python examples/rgt/train.py \
    --task LP \
    --dataset Brazil \
    --gpu 0 \
    --pretrained_model_path Pretrain_ogbn-arxiv_computers_Physics_model \
    --lp_epochs 3
```

> The pretrained model checkpoint (`.pt`) should be placed under `checkpoints/`.

## Notes

- The reproduced results are slightly lower than the original paper, which is expected due to differences in implementation details, random seeds, and environment configurations.
- The `datasets/` directory should contain the pre-trained model checkpoints under `checkpoints/`.
- For custom datasets, place them under `datasets/` and update `data_name` accordingly.
