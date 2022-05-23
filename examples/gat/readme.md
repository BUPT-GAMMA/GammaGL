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
```

| Dataset  | Paper      | Our(pd)      | Our(tf)      |
| -------- | ---------- | ------------ | ------------ |
| cora     | 83.0(±0.7) | 83.54(±0.75) | 83.26(±0.96) |
| pubmed   | 72.5(±0.7) | 72.74(±0.76) | 72.5(±0.65)  |
| citeseer | 79.0(±0.3) | OOM          | OOM          |
