# Learning to Distill Graph Neural Networks (LTD)

- Paper link: [https://doi.org/10.1145/3539597.3570480]
- Author's code repo: [https://github.com/BUPT-GAMMA/LTD].  

# Dataset Statics

## Raw Data Statics

| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

## Largest Connected Component Statics

| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,485   | 10,138  | 7         |
| Citeseer | 2,120   | 7,358   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

# Results

```bash
# available dataset: "cora", "citeseer", "pubmed" 
TL_BACKEND="torch" python ltd_trainer.py --dataset cora --model GCN
TL_BACKEND="torch" python ltd_trainer.py --dataset cora --model GAT
TL_BACKEND="torch" python ltd_trainer.py --dataset citeseer --model GCN 
TL_BACKEND="torch" python ltd_trainer.py --dataset citeseer --model GAT
TL_BACKEND="torch" python ltd_trainer.py --dataset pubmed --model GCN 
TL_BACKEND="torch" python ltd_trainer.py --dataset pubmed --model GAT
```

| Dataset  | GNN  | paper(Largest Connected Component) | Our        | Our(Largest Connected Component) |
| -------- | ---- | ---------------------------------- | ---------- | -------------------------------- |
| cora     | GCN  | 87.21                              | 83.10±0.41 | 84.71±0.44                       |
| cora     | GAT  | 86.56                              | 82.40±1.38 | 84.35±0.42                       |
| citeseer | GCN  | 78.51                              | 72.39±1.09 | 74.69±1.12                       |
| citeseer | GAT  | 77.35                              | 72.85±0.71 | 76.27±0.62                       |
| pubmed   | GCN  | 81.91                              | 79.54±0.70 | 79.54±0.70                       |
| pubmed   | GAT  | 82.74                              | 78.77±0.90 | 78.77±0.90                       |
