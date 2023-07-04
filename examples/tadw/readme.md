# Network Representation Learning with Rich Text Information (TADW)

- Paper link: [https://www.ijcai.org/Proceedings/15/Papers/299.pdf](https://www.ijcai.org/Proceedings/15/Papers/299.pdf)
- Author's code repo: [https://github.com/albertyang33/TADW](https://github.com/albertyang33/TADW). Note that the original code is 
  implemented with MATLAB for the paper.

# Dataset Statics

| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

# Performance
> For all the datasets: The training ratio is 50% for linear SVM.

| Dataset  | Paper(10%) | Paper(20%) | Paper(30%) | Paper(40%) | Paper(50%) | Our(tf)      | Our(th)      | Our(pd)      | Our(ms)      |
| -------- | ---------- | ---------- | ---------- | ---------- | ---------- | ------------ | ------------ | ------------ | ------------ |
| Cora     | 82.4       | 85.0       | 85.6       | 86.0       | 86.7       | 84.43(±0.37) | 84.42(±0.96) | 84.34(±0.64) | 84.08(±0.53) |
| Citeseer | 70.6       | 71.9       | 73.3       | 73.7       | 74.2       | 74.09(±1.11) | 74.41(±0.58) | 73.92(±0.72) | 73.87(±0.57) |

```bash
TL_BACKEND="tensorflow" python tadw_trainer.py --dataset cora --lr 0.2 --n_epoch 100 --embedding_dim 80 --lamda 0.2 --svdft 200 
TL_BACKEND="torch" python tadw_trainer.py --dataset cora --lr 0.2 --n_epoch 100 --embedding_dim 80 --lamda 0.2 --svdft 200
TL_BACKEND="paddle" python tadw_trainer.py --dataset cora --lr 0.2 --n_epoch 100 --embedding_dim 80 --lamda 0.2 --svdft 200
TL_BACKEND="mindspore" python tadw_trainer.py --dataset cora --lr 0.2 --n_epoch 100 --embedding_dim 80 --lamda 0.2 --svdft 200
TL_BACKEND="torch" python tadw_trainer.py --dataset citeseer --lr 0.1 --n_epoch 50 --embedding_dim 500 --lamda 0.5 --svdft 300
TL_BACKEND="tensorflow" python tadw_trainer.py --dataset citeseer --lr 0.1 --n_epoch 50 --embedding_dim 500 --lamda 0.5 --svdft 300
TL_BACKEND="paddle" python tadw_trainer.py --dataset citeseer --lr 0.1 --n_epoch 50 --embedding_dim 500 --lamda 0.5 --svdft 300
TL_BACKEND="mindspore" python tadw_trainer.py --dataset citeseer --lr 0.1 --n_epoch 50 --embedding_dim 500 --lamda 0.5 --svdft 300
```