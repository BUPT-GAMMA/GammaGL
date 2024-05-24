# Graph-less Neural Networks (GLNN)

- Paper link: [https://arxiv.org/pdf/2110.08727](https://arxiv.org/pdf/2110.08727)
- Author's code repo: [https://github.com/snap-research/graphless-neural-networks](https://github.com/snap-research/graphless-neural-networks)

# Dataset Statics
| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |
| Computers| 13,752  | 491,722 | 10        |
| Photo    | 7,650   | 238,162 | 8         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid), [Amazon](https://gammagl.readthedocs.io/en/latest/generated/gammagl.datasets.Amazon.html#gammagl.datasets.Amazon).

# Results

- available dataset: "cora", "citeseer", "pubmed", "computers", "photo"
- avaiable teacher: "SAGE", "GCN", "GAT", "APPNP", "MLP"

```bash
TL_BACKEND="tensorflow" python train_teacher.py --dataset cora --teacher SAGE
TL_BACKEND="tensorflow" python train_student.py --dataset cora --teacher SAGE
TL_BACKEND="torch" python train_teacher.py --dataset cora --teacher SAGE
TL_BACKEND="torch" python train_student.py --dataset cora --teacher SAGE
TL_BACKEND="paddle" python train_teacher.py --dataset cora --teacher SAGE
TL_BACKEND="paddle" python train_student.py --dataset cora --teacher SAGE
TL_BACKEND="mindspore" python train_teacher.py --dataset cora --teacher SAGE
TL_BACKEND="mindspore" python train_student.py --dataset cora --teacher SAGE
```

| Dataset   | Paper      | Our(tf)    | Our(th)    | Our(pd)    | Our(ms)    |
| --------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Cora      | 80.54±1.35 | 80.94±0.31 | 80.84±0.30 | 80.90±0.21 | 81.04±0.30 |
| Citeseer  | 71.77±2.01 | 70.74±0.87 | 71.34±0.55 | 71.18±1.20 | 70.58±1.14 |
| Pubmed    | 75.42±2.31 | 77.90±0.07 | 77.88±0.23 | 77.78±0.19 | 77.78±0.13 |
| Computers | 83.03±1.87 | 81.51±0.60 | 81.73±0.48 | 81.46±0.72 | 81.24±1.27 |
| Photo     | 92.11±1.08 | 92.05±0.56 | 91.92±0.53 | 92.00±0.55 | 91.77±0.91 |

- The model performance is the average of 5 tests