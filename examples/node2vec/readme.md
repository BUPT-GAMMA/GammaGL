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
TL_BACKEND="tensorflow" python node2vec_trainer.py --lr 0.05 --embedding_dim 128 --p 0.5 --q 2.0 --walk_length 20 --window_size 10 --num_walks 10 --n_epoch 100 --dataset cora
TL_BACKEND="tensorflow" python node2vec_trainer.py --lr 0.05 --embedding_dim 128 --p 0.5 --q 2.0 --walk_length 20 --window_size 10 --num_walks 10 --n_epoch 100 --dataset citeseer
TL_BACKEND="tensorflow" python node2vec_trainer.py --lr 0.05 --embedding_dim 128 --p 0.5 --q 2.0 --walk_length 10 --window_size 5 --num_walks 10 --n_epoch 100 --dataset pubmed
TL_BACKEND="torch" python node2vec_trainer.py --lr 0.05 --embedding_dim 64 --p 0.5 --q 2.0 --walk_length 20 --window_size 10 --num_walks 10 --n_epoch 100 --dataset cora
TL_BACKEND="torch" python node2vec_trainer.py --lr 0.05 --embedding_dim 64 --p 0.5 --q 2.0 --walk_length 20 --window_size 10 --num_walks 10 --n_epoch 100 --dataset citeseer
TL_BACKEND="torch" python node2vec_trainer.py --lr 0.05 --embedding_dim 128 --p 0.5 --q 2.0 --walk_length 10 --window_size 5 --num_walks 10 --n_epoch 100 --dataset pubmed
TL_BACKEND="paddle" python node2vec_trainer.py --lr 0.05 --embedding_dim 64 --p 0.5 --q 2.0 --walk_length 20 --window_size 10 --num_walks 10 --n_epoch 100 --dataset cora
TL_BACKEND="paddle" python node2vec_trainer.py --lr 0.05 --embedding_dim 64 --p 0.5 --q 2.0 --walk_length 15 --window_size 10 --num_walks 10 --n_epoch 100 --dataset citeseer
TL_BACKEND="paddle" python node2vec_trainer.py --lr 0.05 --embedding_dim 128 --p 0.5 --q 2.0 --walk_length 10 --window_size 5 --num_walks 10 --n_epoch 100 --dataset pubmed
```



| Dataset  | Our(pd)      | Our(torch)   | Our(tf)      |
| -------- | ------------ | ------------ | ------------ |
| cora     | 68.36(±1.41) | 68.28(±1.18) | 67.63(±1.46) |
| citeseer | 43.41(±1.66) | 43.75(±1.41) | 42.94(±1.80) |
| pubmed   | 59.14(±0.96) | 59.04(±1.33) | 58.69(±1.57) |