# Graph Convolutional Networks (GCN)

- Paper link: [https://arxiv.org/pdf/2401.12564v2](https://arxiv.org/pdf/2401.12564v2)
- Author's code repo: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn). Note that the original code is 
  implemented with Tensorflow for the paper. 

# Dataset Statics

| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

Results
-------

```bash
# available dataset: "cora", "citeseer", "pubmed"
TL_BACKEND="torch" python gcil_trainer.py --dataset cora
TL_BACKEND="torch" python gcil_trainer.py --dataset citeseer
TL_BACKEND="torch" python gcil_trainer.py --dataset pubmed 
```

Ma-F1:
| Dataset  | Paper | Our(th)    |
| -------- | ----- | ---------- |
| cora     | 71.96 | 45.19±0.22 |
| citeseer | 69.10 | 31.94±0.78 |
| pubmed   | 76.32 | 42.98±0.30 |

Mi-F1
|----------|-------|------------|
| cora     | 72.07 | 49.71±0.22 |
| citeseer | 73.70 | 34.80±0.78 |
| pubmed   | 76.88 | 49.69±0.30 |