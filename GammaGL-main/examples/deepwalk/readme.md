# DeepWalk: Online Learning of Social Representations

- Paper link: [https://arxiv.org/abs/1403.6652](https://arxiv.org/abs/1403.6652)

- Author's code repo (in Tensorflow):

    https://github.com/phanein/deepwalk.

- Popular dgl implementation:
    https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/deepwalk.

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
TL_BACKEND="tensorflow" python deepwalk_trainer.py --lr 0.05 --embedding_dim 128 --walk_length 20 --window_size 10 --num_walks 10 --n_epoch 100 --dataset cora
TL_BACKEND="tensorflow" python deepwalk_trainer.py --lr 0.05 --embedding_dim 128 --walk_length 20 --window_size 10 --num_walks 10 --n_epoch 100 --dataset citeseer
TL_BACKEND="tensorflow" python deepwalk_trainer.py --lr 0.05 --embedding_dim 64 --walk_length 10 --window_size 5 --num_walks 10 --n_epoch 100 --dataset pubmed
TL_BACKEND="torch" python deepwalk_trainer.py --lr 0.05 --embedding_dim 128 --walk_length 20 --window_size 10 --num_walks 10 --n_epoch 100 --dataset cora
TL_BACKEND="torch" python deepwalk_trainer.py --lr 0.05 --embedding_dim 128 --walk_length 20 --window_size 10 --num_walks 10 --n_epoch 100 --dataset citeseer
TL_BACKEND="torch" python deepwalk_trainer.py --lr 0.05 --embedding_dim 64 --walk_length 10 --window_size 5 --num_walks 10 --n_epoch 100 --dataset pubmed
TL_BACKEND="paddle" python deepwalk_trainer.py --lr 0.05 --embedding_dim 128 --walk_length 20 --window_size 10 --num_walks 10 --n_epoch 100 --dataset cora
TL_BACKEND="paddle" python deepwalk_trainer.py --lr 0.05 --embedding_dim 128 --walk_length 20 --window_size 10 --num_walks 10 --n_epoch 100 --dataset citeseer
TL_BACKEND="paddle" python deepwalk_trainer.py --lr 0.05 --embedding_dim 64 --walk_length 10 --window_size 5 --num_walks 10 --n_epoch 100 --dataset pubmed
```



| Dataset  | GCN(report) | Our(pd)      | Our(torch)   | Our(tf)      |
| -------- | ----------- | ------------ | ------------ | ------------ |
| cora     | 67.2        | 71.20(±1.13) | 70.84(±1.51) | 70.64(±0.96) |
| citeseer | 43.2        | 46.72(±1.09) | 47.13(±1.41) | 48.04(±1.96) |
| pubmed   | 65.3        | 62.61(±2.41) | 61.74(±2.32) | 62.74(±2.43) |