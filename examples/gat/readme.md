Graph Attention Networks (GAT)
============

- Paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code repo (in Tensorflow):
  [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).
- Popular pytorch implementation:
  [https://github.com/Diego999/pyGAT](https://github.com/Diego999/pyGAT).

Dataset Statics
-------

| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

Results
-------

```bash
TL_BACKEND="paddle" python gat_trainer.py --dataset cora --lr 0.01 --l2_coef 0.01 --drop_rate 0.7
TL_BACKEND="paddle" python gat_trainer.py --dataset citeseer --lr 0.006 --l2_coef 0.04 --drop_rate 0.6
TL_BACKEND="paddle" python gat_trainer.py --dataset pubmed --lr 0.05 --l2_coef 0.0015 --drop_rate 0.6
TL_BACKEND="torch" python gat_trainer.py --dataset cora --lr 0.01 --l2_coef 0.005 --drop_rate 0.7
TL_BACKEND="torch" python gat_trainer.py --dataset citeseer --lr 0.01 --l2_coef 0.01 --drop_rate 0.6
TL_BACKEND="torch" python gat_trainer.py --dataset pubmed --lr 0.01 --l2_coef 0.001 --drop_rate 0.2
TL_BACKEND="tensorflow" python gat_trainer.py --dataset cora --lr 0.01 --l2_coef 0.01 --drop_rate 0.7
TL_BACKEND="tensorflow" python gat_trainer.py --dataset citeseer --lr 0.01 --l2_coef 0.04 --drop_rate 0.7
TL_BACKEND="tensorflow" python gat_trainer.py --dataset pubmed --lr 0.005 --l2_coef 0.003 --drop_rate 0.6
```

| Dataset  | Paper      | Our(pd)      | Our(torch)   | Our(tf)      |
| -------- | ---------- | ------------ | ------------ | ------------ |
| cora     | 83.0(±0.7) | 83.54(±0.75) | 82.44(±0.43) | 83.26(±0.96) |
| citeseer | 72.5(±0.7) | 72.74(±0.76) | 70.94(±0.43) | 72.5(±0.65)  |
| pubmed   | 79.0(±0.3) | 78.82(±0.71) | 78.5(±0.75)  | 78.2(±0.38)  |
