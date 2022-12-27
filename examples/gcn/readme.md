# Graph Convolutional Networks (GCN)

- Paper link: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
- Author's code repo: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn). Note that the original code is 
  implemented with Tensorflow for the paper. 

# Dataset Statics

| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

Results
-------

```bash
# available dataset: "cora", "citeseer", "pubmed"
TL_BACKEND="paddle" python gcn_trainer.py --dataset cora --num_layers 2 --lr 0.01 --l2_coef 0.01 --drop_rate 0.9 
TL_BACKEND="paddle" python gcn_trainer.py --dataset citeseer --num_layers 2 --lr 0.01 --l2_coef 0.01 --drop_rate 0.7 
TL_BACKEND="paddle" python gcn_trainer.py --dataset pubmed --num_layers 2 --lr 0.01 --l2_coef 0.005 --drop_rate 0.6 
TL_BACKEND="tensorflow" python gcn_trainer.py --dataset cora --num_layers 2 --lr 0.005 --l2_coef 0.01 --drop_rate 0.6 
TL_BACKEND="tensorflow" python gcn_trainer.py --dataset citeseer --num_layers 2 --lr 0.01 --l2_coef 0.001 --drop_rate 0.8 
TL_BACKEND="tensorflow" python gcn_trainer.py --dataset pubmed --num_layers 2 --lr 0.01 --l2_coef 0.001 --drop_rate 0.9 
TL_BACKEND="torch" python gcn_trainer.py --dataset cora --num_layers 2 --lr 0.005 --l2_coef 0.01 --drop_rate 0.8 
TL_BACKEND="torch" python gcn_trainer.py --dataset citeseer --num_layers 2 --lr 0.01 --l2_coef 0.01 --drop_rate 0.7 
TL_BACKEND="torch" python gcn_trainer.py --dataset pubmed --num_layers 2 --lr 0.01 --l2_coef 0.002 --drop_rate 0.5 
TL_BACKEND="mindspore" python gcn_trainer.py --dataset cora --num_layers 2 --lr 0.01 --l2_coef 0.01 --drop_rate 0.6
TL_BACKEND="mindspore" python gcn_trainer.py --dataset citeseer --num_layers 2 --lr 0.01 --l2_coef 0.05 --drop_rate 0.7 
TL_BACKEND="mindspore" python gcn_trainer.py --dataset pubmed --num_layers 2 --lr 0.01 --l2_coef 0.01 --drop_rate 0.6 
```

| Dataset  | Paper | Our(pd)    | Our(tf)    | Our(th)    | Our(ms)    |
|----------|-------|------------|------------|------------|------------|
| cora     | 81.5  | 81.83±0.22 | 80.54±1.12 | 81.43±0.17 | 81.50±0.64 |
| citeseer | 70.3  | 70.38±0.78 | 68.34±0.68 | 70.53±0.18 | 71.56±0.14 |
| pubmed   | 79.0  | 78.62±0.30 | 78.28±1.08 | 78.63±0.12 | 79.28±0.17 |
