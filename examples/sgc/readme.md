# Simple Graph Convolution (SGC)

- Paper link: [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153)
- Author's code repo: [https://github.com/Tiiiger/SGC](https://github.com/Tiiiger/SGC). 

Dataset Statics
-------

| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |
Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

Results
-------
```bash
TL_BACKEND="paddle" python sgc_trainer.py --dataset cora --lr 0.2 --n_epoch 250 --iter_K 2 --l2_coef 0.005 --self_loops 1
TL_BACKEND="paddle" python sgc_trainer.py --dataset citeseer --lr 0.01 --n_epoch 250 --iter_K 5 --l2_coef 0.05 --self_loops 1
TL_BACKEND="paddle" python sgc_trainer.py --dataset pubmed --lr 0.1 --n_epoch 200 --iter_K 2 --l2_coef 0.00005 --self_loops 1
TL_BACKEND="tensorflow" python sgc_trainer.py --dataset cora --lr 0.1 --n_epoch 250 --iter_K 5 --l2_coef 0.0005 --self_loops 5
TL_BACKEND="tensorflow" python sgc_trainer.py --dataset citeseer --lr 0.01 --n_epoch 200 --iter_K 15 --l2_coef 0.00005 --self_loops 1
TL_BACKEND="tensorflow" python sgc_trainer.py --dataset pubmed --lr 0.1 --n_epoch 200 --iter_K 15 --l2_coef 0.0005 --self_loops 1
TL_BACKEND="torch" python sgc_trainer.py --dataset cora --lr 0.2 --n_epoch 250 --iter_K 2 --l2_coef 0.005 --self_loops 1
TL_BACKEND="torch" python sgc_trainer.py --dataset citeseer --lr 0.1 --n_epoch 250 --iter_K 2 --l2_coef 0.00005 --self_loops 1
TL_BACKEND="torch" python sgc_trainer.py --dataset pubmed --lr 0.1 --n_epoch 200 --iter_K 2 --l2_coef 0.00005 --self_loops 1
```

| dataset  | paper      | our(tf)      | our(pd)      | our(th)       |
|----------|------------|--------------|--------------|---------------|
| cora     | 81.0(±0)   | 81.45(±0.37) | 81.65(±0.2)  | 81.69(±0.18)  |
| citeseer | 71.9(±0.1) | 69.03(±0.27) | 71.08(±0.04) | 71.63(±0.38)  |
| pubmed   | 78.9(±0)   | 79.1(±0)     | 79.17(±0.05) | 79.16(±0.05)  |

