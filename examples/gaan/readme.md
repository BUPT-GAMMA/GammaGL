Gated Attention Networks  (GaAN)
============

- Paper link: [https://arxiv.org/abs/1803.07294](https://arxiv.org/abs/1803.07294)
- Author's code repo (in mxnet):
  [https://github.com/jennyzhang0215/GaAN](https://github.com/jennyzhang0215/GaAN).

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
TL_BACKEND="tensorflow" python GaAN_trainer.py --dataset cora --n_epoch 200 --lr 0.005 --l2_coef 0.01 --drop_rate 0.1 --heads 8 --v 64 --m 64
TL_BACKEND="tensorflow" python GaAN_trainer.py --dataset citeseer --n_epoch 200 --lr 0.003 --l2_coef 0.005 --drop_rate 0.4 --heads 8 --v 32 --m 32
TL_BACKEND="tensorflow" python GaAN_trainer.py --dataset pubmed --n_epoch 300 --lr 0.005 --l2_coef 0.0005 --drop_rate 0.4 --heads 8 --v 64 --m 64
```

| Dataset  | GAT(Paper) | Our(tf)      |
| -------- | ---------- | ------------ |
| cora     | 83.0(±0.7) | 79.11(±1.78) |
| citeseer | 72.5(±0.7) | 68.15(±3.75) |
| pubmed   | 79.0(±0.3) | 77.29(±1.71) |