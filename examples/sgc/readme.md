Simple Graph Convolution (SGC)
============

- Paper link: [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153)
- Author's code repo: [https://github.com/Tiiiger/SGC](https://github.com/Tiiiger/SGC). 

Dataset Statics
-------
Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

Results
-------
```bash
TL_BACKEND="paddle" python3 sgc_trainer.py --dataset cora --lr 0.2 --n_epoch 250 --iter_K 2 --l2_coef 0.005
TL_BACKEND="paddle" python3 sgc_trainer.py --dataset citeseer --lr 0.2 --n_epoch 250 --iter_K 2 --l2_coef 0.005
TL_BACKEND="paddle" python3 sgc_trainer.py --dataset pubmed --lr 0.2 --n_epoch 250 --iter_K 2 --l2_coef 0.005
```

| dataset  | paper      | our(tf)      | our(pd)       |
|----------|------------|--------------|---------------|
| cora     | 81.0(±0)   | 81.45(±0.37) | 81.65(±0.2)   |
| citeseer | 71.9(±0.1) | 69.03(±0.27) | 71.08(±0.04)  |
| pubmed   | 78.9(±0)   | 79.1(±0)     | 79.71(±0.05)  |

