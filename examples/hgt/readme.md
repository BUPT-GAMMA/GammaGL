

# Heterogeneous Graph Transformer (HGT)

- Paper link: [https://arxiv.org/pdf/2003.01332](https://arxiv.org/pdf/2003.01332)


# Dataset Statics

| Dataset | # Nodes | # Node Types | # Edges | # Edge Types | Target | # Classes |
| ------- | ------- | ------------ | ------- | ------------ | ------ | --------- |
| DBLP    | 26,128  | 4            | 239,566 | 6            | author | 4         |
| IMDB    | 21,420  | 4            | 86,642  | 6            | movie  | 5         |

DBLP dataset refer to [HGBDataset](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.HGBDataset).

IMDBdataset refer to [IMDB](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.IMDB).

Results
-------

```bash
# available dataset: "DBLP", "IMDB"
TL_BACKEND="paddle" python hgt_trainer.py --dataset DBLP --lr 0.0001 --n_epoch 200 --hidden_dim 1024 --l2_coef 1e-6 --heads 4 --drop_rate 0.9
TL_BACKEND="torch" python hgt_trainer.py --dataset DBLP --lr 0.0001 --n_epoch 200 --hidden_dim 1024 --l2_coef 5e-6 --heads 4 --drop_rate 0.9
TL_BACKEND="paddle" python hgt_trainer.py --dataset IMDB --lr 0.0001 --n_epoch 200 --hidden_dim 1024 --l2_coef 1e-6 --heads 4 --drop_rate 0.9
TL_BACKEND="torch" python hgt_trainer.py --dataset IMDB --lr 0.0001 --n_epoch 150 --hidden_dim 1024 --l2_coef 1e-6 --heads 4 --drop_rate 0.5
TL_BACKEND="tensorflow" python hgt_trainer.py --dataset IMDB --lr 0.0001 --n_epoch 200 --hidden_dim 1024 --l2_coef 5e-6 --heads 4 --drop_rate 0.9
```



| Dataset | Paper      | Our(pd)      | Our(tf)      | Our(torch)   |
| ------- | ---------- | ------------ | ------------ | ------------ |
| DBLP    | 93.01±0.23 | 92.4(±0.92)  |              | 90.89(±1.08) |
| IMDB    | 63.00±1.19 | 54.51(±1.99) | 55.98(±2.09) | 54.93(±1.34) |

The experimental results under IMDB dataset are consistent with [PyG](https://github.com/pyg-team/pytorch_geometric).
