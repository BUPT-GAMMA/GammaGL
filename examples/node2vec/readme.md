# node2vec: Scalable Feature Learning for Networks

- Paper link: [https://arxiv.org/abs/1607.00653](https://arxiv.org/abs/1607.00653)

- Author's code repo (in Tensorflow):

    https://github.com/aditya-grover/node2vec.

- Popular pytorch implementation:
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py.

Dataset Statics
-------

| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

## Results

```bash
TL_BACKEND="tensorflow" python node2vec_trainer.py --lr 0.05 --embedding_dim 128 --p 4.0 --q 0.25 --walk_length 20 --window_size 10 --num_walks 10 --n_epoch 100 --dataset cora
TL_BACKEND="tensorflow" python node2vec_trainer.py --lr 0.05 --embedding_dim 128 --p 4.0 --q 0.25 --walk_length 20 --window_size 10 --num_walks 10 --n_epoch 100 --dataset citeseer
TL_BACKEND="tensorflow" python node2vec_trainer.py --lr 0.05 --embedding_dim 64 --p 2.0 --q 0.5 --walk_length 10 --window_size 5 --num_walks 10 --n_epoch 100 --dataset pubmed
TL_BACKEND="torch" python node2vec_trainer.py --lr 0.05 --embedding_dim 128 --p 4.0 --q 0.25 --walk_length 20 --window_size 10 --num_walks 10 --n_epoch 100 --dataset cora
TL_BACKEND="torch" python node2vec_trainer.py --lr 0.05 --embedding_dim 128 --p 4.0 --q 0.25 --walk_length 20 --window_size 10 --num_walks 10 --n_epoch 100 --dataset citeseer
TL_BACKEND="torch" python node2vec_trainer.py --lr 0.05 --embedding_dim 64 --p 4.0 --q 0.25 --walk_length 10 --window_size 5 --num_walks 10 --n_epoch 100 --dataset pubmed
TL_BACKEND="paddle" python node2vec_trainer.py --lr 0.05 --embedding_dim 128 --p 4.0 --q 0.25 --walk_length 20 --window_size 10 --num_walks 10 --n_epoch 100 --dataset cora
TL_BACKEND="paddle" python node2vec_trainer.py --lr 0.05 --embedding_dim 128 --p 4.0 --q 0.25 --walk_length 20 --window_size 10 --num_walks 10 --n_epoch 100 --dataset citeseer
TL_BACKEND="paddle" python node2vec_trainer.py --lr 0.05 --embedding_dim 64 --p 2.0 --q 0.5 --walk_length 10 --window_size 5 --num_walks 10 --n_epoch 100 --dataset pubmed
```

| Dataset  | pyg          | Our(pd)      | Our(torch)   | Our(tf)      |
| -------- | ------------ | ------------ | ------------ | ------------ |
| cora     | 71.06(±0.62) | 72.33(±1.65) | 71.91(±0.74) | 71.61(±0.96) |
| citeseer | 48.20(±1.01) | 48.55(±1.32) | 48.76(±0.91) | 48.23(±1.35) |
| pubmed   | 61.60(±1.83) | 62.28(±1.33) | 61.82(±2.48) | 62.89(±2.45) |

Note: The hyperparameter settings for the results in pyg are the same as in GammaGL.