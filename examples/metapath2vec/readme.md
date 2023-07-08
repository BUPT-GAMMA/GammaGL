# metapath2vec: Scalable Representation Learning for
Heterogeneous Networks
for Recommendation

- Paper link: [https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf)

- Author's code repo (in Tensorflow):

    https://github.com/apple2373/metapath2vec.


Dataset Statics
-------

| Dataset | # Nodes   | # Node Types | # Edges    | # Edge Types | Target | # Classes |
| ------- | --------- | ------------ | ---------- | ------------ | ------ | --------- |
| AMiner  | 4,891,819 | 3            | 25,036,020 | 4            | author | 8         |
| DBLP    | 26,128    | 4            | 239,566    | 6            | author | 4         |
| IMDB    | 11,616    | 3            | 34,212     | 4            | movie  | 3         |

## Results

```bash
TL_BACKEND="torch" python metapath2vec_trainer_aminer.py --lr 0.1 --embedding_dim 16 --walk_length 60 --window_size 3 --num_walks 600 --n_epoch 5 --num_negative_samples 6 --batch_size 128 --train_ratio 0.5 --dataset aminer
TL_BACKEND="torch" python metapath2vec_trainer_imdb&dblp.py --lr 0.01 --embedding_dim 16 --walk_length 50 --window_size 7 --num_walks 5 --n_epoch 50 --num_negative_samples 5 --batch_size 128 --dataset imdb
TL_BACKEND="torch" python metapath2vec_trainer_imdb&dblp.py --lr 0.01 --embedding_dim 16 --walk_length 50 --window_size 7 --num_walks 5 --n_epoch 50 --num_negative_samples 5 --batch_size 128 --dataset dblp
TL_BACKEND="tensorflow" python metapath2vec_trainer_aminer.py --lr 0.1 --embedding_dim 16 --walk_length 60 --window_size 3 --num_walks 500 --n_epoch 5 --num_negative_samples 6 --batch_size 128 --train_ratio 0.5 --dataset aminer
TL_BACKEND="tensorflow" python metapath2vec_trainer_imdb&dblp.py --lr 0.01 --embedding_dim 16 --walk_length 50 --window_size 7 --num_walks 5 --n_epoch 50 --num_negative_samples 5 --batch_size 128 --dataset imdb
TL_BACKEND="tensorflow" python metapath2vec_trainer_imdb&dblp.py --lr 0.01 --embedding_dim 16 --walk_length 50 --window_size 7 --num_walks 5 --n_epoch 50 --num_negative_samples 5 --batch_size 128 --dataset dblp
TL_BACKEND="paddle" python metapath2vec_trainer_aminer.py --lr 0.1 --embedding_dim 16 --walk_length 60 --window_size 3 --num_walks 600 --n_epoch 5 --num_negative_samples 6 --batch_size 128 --train_ratio 0.5 --dataset aminer
TL_BACKEND="paddle" python metapath2vec_trainer_imdb&dblp.py --lr 0.01 --embedding_dim 16 --walk_length 50 --window_size 7 --num_walks 5 --n_epoch 50 --num_negative_samples 5 --batch_size 128 --dataset imdb
TL_BACKEND="paddle" python metapath2vec_trainer_imdb&dblp.py --lr 0.01 --embedding_dim 16 --walk_length 50 --window_size 7 --num_walks 5 --n_epoch 50 --num_negative_samples 5 --batch_size 128 --dataset dblp
```



| Dataset | HAN(report) | Our(th)      | Our(tf)      | Our(pd)      |
| ------- | ----------- | ------------ | ------------ | ------------ |
| AMiner  | 84.27(pyg)  | 84.47(±0.57) | 83.54(±1.15) | 84.05(±1.43) |
| IMDB    | 45.65       | 51.80(±0.43) | 51.54(±1.23) | 51.24(±1.42) |
| DBLP    | 91.53       | 91.76(±0.34) | 91.42(±0.22) | 91.56(±1.56) |

Note: The hyperparameter settings for the results in the AMiner dataset is the same as in GammaGL.