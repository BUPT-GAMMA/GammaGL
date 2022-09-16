# DropEdge

- Paper link: [https://arxiv.org/abs/1907.10903](https://arxiv.org/abs/1907.10903)
- Author's code repo: [ https://github.com/DropEdge/DropEdge]( https://github.com/DropEdge/DropEdge).
# Dataset Statics

| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |
Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

Results(cora & 4 layer GCN)
-------
```bash
# available dataset: "cora", "citeseer", "pubmed"
TL_BACKEND="paddle" python dropedge_trainer.py --dataset cora --lr 0.01 --l2_coef 0.0005 --drop_rate 0.8 --n_layers 4 --sampling_percent 0.7
TL_BACKEND="paddle" python dropedge_trainer.py --dataset cora --lr 0.01 --l2_coef 0.0005 --drop_rate 0.8 --n_layers 4 --sampling_percent 1
TL_BACKEND="tensorflow" python dropedge_trainer.py --dataset cora --lr 0.01 --l2_coef 0.0005 --drop_rate 0.8 --n_layers 4 --sampling_percent 0.7
TL_BACKEND="tensorflow" python dropedge_trainer.py --dataset cora --lr 0.01 --l2_coef 0.0005 --drop_rate 0.8 --n_layers 4 --sampling_percent 1
TL_BACKEND="torch" python dropedge_trainer.py --dataset cora --lr 0.01 --l2_coef 0.0005 --drop_rate 0.8 --n_layers 4 --sampling_percent 0.7
TL_BACKEND="torch" python dropedge_trainer.py --dataset cora --lr 0.01 --l2_coef 0.0005 --drop_rate 0.8 --n_layers 4 --sampling_percent 1
```
| Method   | Paper  | Our(tf)    | Our(th)    | Our(pd)   |
|----------|--------|------------|------------|-----------|
| Original | 80.40  | 80.42±1.2  | 81.16±1.14 | 80.44±0.66|
| DropEdge | 82.00  | 81.18±2.22 | 81.94±1.56 | 81.20±1.4 |



* The model performance is the average of 5 tests