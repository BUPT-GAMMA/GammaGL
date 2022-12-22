# Graph Structure Estimation Neural Networks (GEN)

- Paper link: [Graph Structure Estimation Neural Networks | Proceedings of the Web Conference 2021](https://dl.acm.org/doi/10.1145/3442381.3449952)
- Author's code repo: https://github.com/BUPT-GAMMA/Graph-Structure-Estimation-Neural-Networks

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
TL_BACKEND="paddle" python gen_trainer.py --dataset cora
TL_BACKEND="paddle" python gen_trainer.py --dataset citeseer
TL_BACKEND="tensorflow" python gen_trainer.py --dataset cora 
TL_BACKEND="tensorflow" python gen_trainer.py --dataset citeseer
TL_BACKEND="torch" python gen_trainer.py --dataset cora
TL_BACKEND="torch" python gen_trainer.py --dataset citeseer
```

| Dataset  | Paper    | Our(pd) | Our(tf) | Our(th) |
| -------- | -------- | ------- | ------- | ------- |
| cora     | 83.6±0.4 |         | 82.60   | 82.50   |
| citeseer | 73.8±0.6 |         | 71.50   | 69.50   |
