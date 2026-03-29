# Graph Convolutional Networks (GCN)

- Paper link: [https://arxiv.org/abs/2009.03509](https://arxiv.org/abs/2009.03509)

# Dataset Statics

| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

Results
-------

```bash
# available dataset: "cora", "citeseer", "pubmed"
TL_BACKEND="tensorflow" python unimp_trainer.py --dataset cora 
TL_BACKEND="tensorflow" python unimp_trainer.py --dataset citeseer 
TL_BACKEND="tensorflow" python unimp_trainer.py --dataset pubmed 
TL_BACKEND="torch" python unimp_trainer.py --dataset cora 
TL_BACKEND="torch" python unimp_trainer.py --dataset citeseer
TL_BACKEND="torch" python unimp_trainer.py --dataset pubmed
```

| Dataset  | Our(tf)    | Our(torch)    |
|----------|------------|------------|
| cora     | 83.10±1.12 | 82.30±0.67 |
| citeseer | 79.90±0.68 | 78.53±0.18 |
| pubmed   | 74.10±1.08 | 73.63±0.12 |
