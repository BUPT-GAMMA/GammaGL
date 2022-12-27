Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing (MixHop)
============

- Paper link: [https://arxiv.org/abs/1905.00067](https://arxiv.org/abs/1905.00067)

- Author's code repo (in Tensorflow):

    https://github.com/samihaija/mixhop.

- Popular pytorch implementation:
    https://github.com/dmlc/dgl/tree/master/examples/pytorch/mixhop.

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
TL_BACKEND="paddle" python mixhop_trainer.py --dataset cora --lr 0.001 --l2_coef 1e-2 --drop_rate 0.5 --hidden_dim 64 --n_epoch 200
TL_BACKEND="paddle" python mixhop_trainer.py --dataset citeseer --lr 0.001 --l2_coef 5e-2 --drop_rate 0.4 --hidden_dim 64 --n_epoch 200
TL_BACKEND="paddle" python mixhop_trainer.py --dataset pubmed --lr 0.0005 --l2_coef 5e-4 --drop_rate 0.5 --hidden_dim 256 --n_epoch 200
TL_BACKEND="torch" python mixhop_trainer.py --dataset cora --lr 0.0005 --l2_coef 5e-4 --drop_rate 0.4 --hidden_dim 64 --n_epoch 200
TL_BACKEND="torch" python mixhop_trainer.py --dataset citeseer --lr 0.001 --l2_coef 5e-2 --drop_rate 0.4 --hidden_dim 64 --n_epoch 200
TL_BACKEND="torch" python mixhop_trainer.py --dataset pubmed --lr 0.001 --l2_coef 1e-3 --drop_rate 0.5 --hidden_dim 256 --n_epoch 200
TL_BACKEND="tensorflow" python mixhop_trainer.py --dataset cora --lr 0.005 --l2_coef 5e-3 --drop_rate 0.5 --hidden_dim 64 --n_epoch 200
TL_BACKEND="tensorflow" python mixhop_trainer.py --dataset citeseer --lr 0.001 --l2_coef 5e-2 --drop_rate 0.4 --hidden_dim 64 --n_epoch 200
TL_BACKEND="tensorflow" python mixhop_trainer.py --dataset pubmed --lr 0.001 --l2_coef 1e-3 --drop_rate 0.5 --hidden_dim 256 --n_epoch 200
```

| Dataset  | Paper       | Our(pd)     | Our(torch)  | Our(tf)       |
| -------- | ----------- | ----------- | ----------- | ------------- |
| cora     | 81.8(±0.62) | 81.6(±0.57) | 81.7(±1.1)  | 82.1(±0.90)   |
| citeseer | 71.4(±0.81) | 71.4(±0.84) | 71.3(±0.54) | 71.5(±0.53)   |
| pubmed   | 80.0(±1.1)  | 78.7(±0.46) | 79.0(±0.39) | 78.7(±0.27)   |