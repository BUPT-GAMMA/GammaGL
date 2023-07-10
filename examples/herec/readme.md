# Heterogeneous Information Network Embedding for Recommendation
- Paper link: [https://arxiv.org/pdf/1711.10730.pdf](https://arxiv.org/pdf/1711.10730.pdf)

- Author's code repo:

    https://github.com/librahu/HERec


Dataset Statics
-------

| Dataset | # Nodes   | # Node Types | # Edges    | # Edge Types | Target | # Classes |
| ------- | --------- | ------------ | ---------- | ------------ | ------ | --------- |
| AMiner  | 4,891,819 | 3            | 25,036,020 | 4            | author | 8         |
| DBLP    | 26,128    | 4            | 239,566    | 6            | author | 4         |
| IMDB    | 11,616    | 3            | 34,212     | 4            | movie  | 3         |

## Results

```bash
TL_BACKEND="torch" python herec_trainer_aminer.py --lr 0.1 --embedding_dim 32 --walk_length 60 --window_size 3 --num_walks 800 --n_epoch 5 --num_negative_samples 10 --batch_size 128 --train_ratio 0.5 --dataset aminer
TL_BACKEND="torch" python herec_trainer_imdb_dblp.py --lr 0.01 --embedding_dim 16 --walk_length 50 --window_size 7 --num_walks 5 --n_epoch 50 --num_negative_samples 5 --batch_size 128 --dataset imdb
TL_BACKEND="torch" python herec_trainer_imdb_dblp.py --lr 0.01 --embedding_dim 64 --walk_length 100 --window_size 5 --num_walks 10 --n_epoch 50 --num_negative_samples 5 --batch_size 128 --dataset dblp
TL_BACKEND="tensorflow" python herec_trainer_aminer.py --lr 0.1 --embedding_dim 32 --walk_length 80 --window_size 3 --num_walks 800 --n_epoch 5 --num_negative_samples 15 --batch_size 128 --train_ratio 0.5 --dataset aminer
TL_BACKEND="tensorflow" python herec_trainer_imdb_dblp.py --lr 0.01 --embedding_dim 16 --walk_length 50 --window_size 7 --num_walks 20 --n_epoch 50 --num_negative_samples 5 --batch_size 128 --dataset imdb
TL_BACKEND="tensorflow" python herec_trainer_imdb_dblp.py --lr 0.01 --embedding_dim 64 --walk_length 100 --window_size 5 --num_walks 10 --n_epoch 50 --num_negative_samples 5 --batch_size 128 --dataset dblp
TL_BACKEND="paddle" python herec_trainer_aminer.py --lr 0.1 --embedding_dim 32 --walk_length 200 --window_size 5 --num_walks 800 --n_epoch 5 --num_negative_samples 20 --batch_size 128 --train_ratio 0.5 --dataset aminer
TL_BACKEND="paddle" python herec_trainer_imdb_dblp.py --lr 0.01 --embedding_dim 16 --walk_length 50 --window_size 7 --num_walks 5 --n_epoch 50 --num_negative_samples 5 --batch_size 128 --dataset imdb
TL_BACKEND="paddle" python herec_trainer_imdb_dblp.py --lr 0.01 --embedding_dim 64 --walk_length 100 --window_size 5 --num_walks 10 --n_epoch 50 --num_negative_samples 5 --batch_size 128 --dataset dblp
```



| Dataset | HAN(report) | Our(th)      | Our(tf)      | Our(pd)      |
| ------- | ----------- | ------------ | ------------ | ------------ |
| AMiner  | --          | 82.11(±0.63) | 80.28(±0.84) | 81.91(±1.12) |
| IMDB    | 45.81       | 51.92(±0.96) | 51.78(±0.69) | 51.86(±1.25) |
| DBLP    | 92.69       | 93.39(±0.38) | 93.12(±0.69) | 93.14(±0.85) |