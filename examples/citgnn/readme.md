# Learning Invariant Representations of Graph Neural Networks via Cluster Generalization (CITGNN)

- Paper link: [https://arxiv.org/pdf/2403.03599]
- Author's code repo: https://github.com/BUPT-GAMMA/CITGNN

# Dataset Statics

| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

# Results

- Available dataset: "cora", "citeseer", "pubmed"
- Available gnn: "GCN", "GAT", "GCNII"

```bash
# available dataset: "cora", "citeseer", "pubmed"
python citgnn_trainer.py --gnn gcn --dataset cora --ss 0.5
python citgnn_trainer.py --gnn gcn --dataset citeseer --ss 0.5
python citgnn_trainer.py --gnn gcn --dataset pubmed --ss 0.5

python citgnn_trainer.py --gnn gat --dataset cora --ss 0.5
python citgnn_trainer.py --gnn gat --dataset citeseer --ss 0.5
python citgnn_trainer.py --gnn gat --dataset pubmed --ss 0.5

python citgnn_trainer.py --gnn gcnii --dataset cora --ss 0.5
python citgnn_trainer.py --gnn gcnii --dataset citeseer --ss 0.5
python citgnn_trainer.py --gnn gcnii --dataset pubmed --ss 0.5
```

ADD-0.5

|            | Paper      |            |            |            |            |            |
| ---        | ---        | ---        | ---        | ---        | ---        | ---        |
| Method     | Cora       |            | Citeseer   |            | Pubmed     |            |
|            | Acc        | Macro-f1   | Acc        | Macro-f1   | Acc        | Macro-f1   |
| CIT-GCN    | 76.98±0.49 | 75.88±0.44 | 67.65±0.44 | 64.42±0.10 | 73.76±0.40 | 72.94±0.30 |
| CIT-GAT    | 77.23±0.42 | 76.26±0.28 | 66.33±0.24 | 63.07±0.37 | 72.50±0.74 | 71.57±0.82 |
| CIT-GCNII  | 78.28±0.88 | 75.82±0.73 | 66.12±0.97 | 63.17±0.85 | 75.95±0.63 | 75.47±0.76 |

|            | Our        |            |            |            |            |            |
| ---        | ---        | ---        | ---        | ---        | ---        | ---        |
| Method     | Cora       |            | Citeseer   |            | Pubmed     |            |
|            | Acc        | Macro-f1   | Acc        | Macro-f1   | Acc        | Macro-f1   |
| CIT-GCN    | 78.43±1.15 | 76.95±1.26 | 65.78±0.91 | 61.12±1.29 | 77.96±0.72 | 75.98±1.01 |
| CIT-GAT    | 75.67±0.91 | 74.23±1.21 | 60.68±0.73 | 56.19±0.97 | 74.77±0.78 | 72.91±1.15 |
| CIT-GCNII  | 77.49±1.06 | 76.32±1.19 | 65.94±1.04 | 61.97±1.30 | 76.27±0.49 | 74.56±0.77 |
