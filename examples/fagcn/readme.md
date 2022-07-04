# Frequency Adaptation Graph Convolutional Networks (FAGCN)

- Paper link: [https://arxiv.org/abs/2101.00797](https://arxiv.org/abs/2101.00797)
- Author's code repo: [https://github.com/bdy9527/FAGCN](https://github.com/bdy9527/FAGCN)

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
TL_BACKEND="tensorflow" python fagcn_trainer.py --dataset cora --lr 0.01 --l2_coef 0.0005 --drop_rate 0.4 --hidden_dim 16 --eps 0.3 --num_layers 5
TL_BACKEND="tensorflow" python fagcn_trainer.py --dataset citeseer --lr 0.01 --l2_coef 0.0005 --drop_rate 0.4 --hidden_dim 16 --eps 0.4 --num_layers 4
TL_BACKEND="tensorflow" python fagcn_trainer.py --dataset pubmed --lr 0.01 --l2_coef 0.001 --drop_rate 0.4 --hidden_dim 16 --eps 0.1 --num_layers 6
TL_BACKEND="paddle" python fagcn_trainer.py --dataset cora --lr 0.01 --l2_coef 0.001 --drop_rate 0.6 --hidden_dim 16 --eps 0.2 --num_layers 3
TL_BACKEND="paddle" python fagcn_trainer.py --dataset citeseer --lr 0.01 --l2_coef 0.001 --drop_rate 0.6 --hidden_dim 16 --eps 0.2 --num_layers 5
TL_BACKEND="paddle" python fagcn_trainer.py --dataset pubmed --lr 0.01 --l2_coef 0.001 --drop_rate 0.4 --hidden_dim 16 --eps 0.2 --num_layers 6
TL_BACKEND="torch" python fagcn_trainer.py --dataset cora --lr 0.005 --l2_coef 0.0005 --drop_rate 0.4 --hidden_dim 16 --eps 0.3 --num_layers 5
TL_BACKEND="torch" python fagcn_trainer.py --dataset citeseer --lr 0.005 --l2_coef 0.001 --drop_rate 0.4 --hidden_dim 16 --eps 0.3 --num_layers 3
TL_BACKEND="torch" python fagcn_trainer.py --dataset pubmed --lr 0.005 --l2_coef 0.001 --drop_rate 0.4 --hidden_dim 16 --eps 0.5 --num_layers 6
```
| Dataset  | Paper    | Our(tf)  | Our(pd)  | Our(torch) |
|----------|----------|----------|----------|------------|
| cora     | 84.1±0.5 | 83.1±0.4 | 82.1±0.4 | 78.1±0.7   |
| citeseer | 72.7±0.8 | 68.3±0.8 | 68.2±0.8 | 65.3±1.3   |
| pubmed   | 79.4±0.3 | 79.2±0.1 | 79.7±0.3 | 77.9±0.8   |
