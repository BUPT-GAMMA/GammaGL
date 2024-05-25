Understanding GNN Computational Graph: A Coordinated Computation, IO, and Memory Perspective (FusedGAT)
============

- Paper link: [https://arxiv.org/abs/2110.09524](https://arxiv.org/abs/2110.09524)
- Author's code repo (in PyTorch):
  [https://github.com/dgSPARSE/dgNN](https://github.com/dgSPARSE/dgNN).

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
TL_BACKEND="torch" python fusedgat_trainer.py --dataset cora --lr 0.01 --l2_coef 0.005 --drop_rate 0.7
TL_BACKEND="torch" python fusedgat_trainer.py --dataset citeseer --lr 0.01 --l2_coef 0.01 --drop_rate 0.6
TL_BACKEND="torch" python fusedgat_trainer.py --dataset pubmed --lr 0.01 --l2_coef 0.001 --drop_rate 0.2

TL_BACKEND="torch" python test_fusedgat.py --dataset cora --lr 0.01 --l2_coef 0.005 --drop_rate 0.7
TL_BACKEND="torch" python test_fusedgat.py --dataset citeseer --lr 0.01 --l2_coef 0.01 --drop_rate 0.6
TL_BACKEND="torch" python test_fusedgat.py --dataset pubmed --lr 0.01 --l2_coef 0.001 --drop_rate 0.2

TL_BACKEND="torch" python test_gat.py --dataset cora --lr 0.01 --l2_coef 0.005 --drop_rate 0.7
TL_BACKEND="torch" python test_gat.py --dataset citeseer --lr 0.01 --l2_coef 0.01 --drop_rate 0.6
TL_BACKEND="torch" python test_gat.py --dataset pubmed --lr 0.01 --l2_coef 0.001 --drop_rate 0.2
```

| Dataset  | Our(torch)   |
| -------- | ------------ |
| cora     |    79.68     |
| citeseer |    66.20     |
| pubmed   |    76.96     |


| Dataset  | Metric     | GAT          | FusedGAT     |
| -------- | ---------- | ------------ | ------------ |
| cora     | train      |    20.37     |    10.09     |
| cora     | infer      |     4.04     |     2.11     |
| cora     | memory     |     359      |     341      |
| citeseer | train      |    21.16     |     9.99     |
| citeseer | infer      |     4.28     |     2.19     |
| citeseer | memory     |     421      |     451      |
| pubmed   | train      |    20.44     |     9.05     |
| pubmed   | infer      |     4.21     |     2.11     |
| pubmed   | memory     |     505      |     427      |

train : ms / epoch

infer : ms / epoch

memory: MB