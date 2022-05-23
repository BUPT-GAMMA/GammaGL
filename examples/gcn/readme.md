# Graph Convolutional Networks (GCN)

- Paper link: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
- Author's code repo: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn). Note that the original code is 
  implemented with Tensorflow for the paper. 

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
TL_BACKEND="paddle" python gcn_trainer.py --dataset cora --lr 0.01 --l2_coef 0.005 --drop_rate 0.9
TL_BACKEND="paddle" python gcn_trainer.py --dataset citeseer --lr 0.01 --l2_coef 0.01 --drop_rate 0.7
TL_BACKEND="paddle" python gcn_trainer.py --dataset pubmed --lr 0.01 --l2_coef 0.005 --drop_rate 0.6
```

| Dataset  | Paper | Our(pd)    | Our(tf)    |
| -------- | ----- | ---------- | ---------- |
| cora     | 81.5  | 81.83±0.22 | 80.54±1.12 |
| citeseer | 70.3  | 70.38±0.78 | 68.34±0.68 |
| pubmed   | 79.0  | 78.62±0.30 | 78.28±1.08 |
