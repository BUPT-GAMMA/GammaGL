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
python citgnn_trainer.py --gnn gcn --dataset cora --lr 0.005 --l2_coef 0.01 --droprate 0.8 
python citgnn_trainer.py --gnn gcn --dataset citeseer --lr 0.01 --l2_coef 0.01 --droprate 0.7 
python citgnn_trainer.py --gnn gcn --dataset pubmed --lr 0.01 --l2_coef 0.002 --droprate 0.5

python citgnn_trainer.py --gnn gat --dataset cora --lr 0.005 --l2_coef 0.005 --droprate 0.5
python citgnn_trainer.py --gnn gat --dataset citeseer --lr 0.01 --l2_coef 0.005 --droprate 0.5
python citgnn_trainer.py --gnn gat --dataset pubmed --lr 0.01 --l2_coef 0.001 --droprate 0.2

python citgnn_trainer.py --gnn gcnii --dataset cora --lr 0.01 --l2_coef 0.001 --droprate 0.3
python citgnn_trainer.py --gnn gcnii --dataset citeseer --lr 0.01 --l2_coef 0.001 --droprate 0.4
python citgnn_trainer.py --gnn gcnii --dataset pubmed --lr 0.01 --l2_coef 0.001 --droprate 0.6
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
| CIT-GCN    | 77.52±1.08 | 76.49±0.49 | 65.78±0.91 | 62.60±1.17 | 72.42±0.25 | 71.65±0.44 |
| CIT-GAT    | 75.84±0.56 | 74.81±0.66 | 63.41±1.28 | 59.98±1.42 | 71.80±0.64 | 70.78±0.69 |
| CIT-GCNII  | 80.30±1.06 | 78.44±1.18 | 65.94±1.04 | 62.67±0.80 | 76.27±0.49 | 75.30±0.67 |

