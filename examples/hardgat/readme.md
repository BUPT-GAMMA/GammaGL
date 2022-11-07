Graph Hard Attention Networks (HardGAT)
============

- Paper link: [https://dl.acm.org/doi/pdf/10.1145/3292500.3330897](https://dl.acm.org/doi/pdf/10.1145/3292500.3330897)
- Popular pytorch implementation:
  [https://github.com/dmlc/dgl/tree/master/examples/pytorch/hardgat](https://github.com/dmlc/dgl/tree/master/examples/pytorch/hardgat)

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
TL_BACKEND="paddle" python gat_trainer.py --dataset cora --lr 0.01 --l2_coef 0.004 --k 8 --drop_rate 0.7
TL_BACKEND="paddle" python gat_trainer.py --dataset citeseer --lr 0.01 --l2_coef 0.004 --k 8 --drop_rate 0.6
TL_BACKEND="paddle" python gat_trainer.py --dataset pubmed --lr 0.01 --l2_coef 0.0015 --k 8 --drop_rate 0.6

TL_BACKEND="torch" python gat_trainer.py --dataset cora --lr 0.01 --l2_coef 0.005 --k 8 --drop_rate 0.7 
TL_BACKEND="torch" python gat_trainer.py --dataset citeseer --lr 0.005 --l2_coef 0.003 --k 8 --drop_rate 0.55
TL_BACKEND="torch" python gat_trainer.py --dataset pubmed --lr 0.01 --l2_coef 0.0015 --k 8 --drop_rate 0.6

TL_BACKEND="tensorflow" python gat_trainer.py --dataset cora --lr 0.03 --l2_coef 0.004 --k 8 --drop_rate 0.7 
TL_BACKEND="tensorflow" python gat_trainer.py --dataset citeseer --lr 0.01 --l2_coef 0.006 --k 8 --drop_rate 0.7
TL_BACKEND="tensorflow" python gat_trainer.py --dataset pubmed --lr 0.01 --l2_coef 0.0015 --k 8 --drop_rate 0.6
```

| Dataset  |   Paper    |   Our(pd)    |  Our(torch)  |   Our(tf)    |
| :------: | :--------: | :----------: | :----------: | :----------: |
|   cora   | 83.5(±0.7) | 83.50(±0.75) | 82.55(±0.40) | 83.80(±0.96) |
| citeseer | 72.7(±0.6) | 72.50(±0.76) | 71.00(±0.45) | 72.30(±0.65) |
|  pubmed  | 79.2(±0.4) | 78.90(±0.76) | 78.53(±0.85) | 78.32(±0.33) |
