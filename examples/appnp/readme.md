Predict then Propagate: Graph Neural Networks meet Personalized PageRank (APPNP)
============

- Paper link: [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997)

- Author's code repo:https://github.com/gasteigerjo/ppnp). 

> This example does not contain the implementation of PPNP.

Dataset Statics
-------
| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |
| cora-ml  | 2810    | 7981    | 7         |

Results
-------
```bash
TL_BACKEND="paddle" python appnp_trainer.py --dataset cora --lr 0.15 --n_epoch 200 --hidden_dim 64 --drop_rate 0.4 --l2_coef 0.02 --iter_K 6 --self_loops 0
TL_BACKEND="paddle" python appnp_trainer.py --dataset pubmed --lr 0.2 --n_epoch 250 --hidden_dim 64 --drop_rate 0.6 --l2_coef 0.001 --iter_K 10 --self_loops 2
TL_BACKEND="paddle" python appnp_trainer.py --dataset citeseer --lr 0.03 --n_epoch 500 --hidden_dim 32 --alpha 0.1 --drop_rate 0.4 --l2_coef 2e-3 --iter_K 10 --self_loops 1
```
| dataset  | paper        | our(pd)     | our(tf)     |
|----------|--------------|-------------|-------------|
| cora     |              | 82.9(±0.56) | 76.3(±0.45) |
| citeseer | 75.83(±0.27) | 70.8(±0.00) | 65.7(±0.15) |
| pubmed   | 79.73(±0.31) | 79.8(±0.67) | 75.5(±0)    |
| cora-ml  | 85.29(±0.25) |             |             |

