# CoBFormer: Less is More - On the Over-Globalizing Problem in Graph Transformers

- Paper link: [https://arxiv.org/abs/2405.14786](https://arxiv.org/abs/2405.14786)
- Author's code repo: [https://github.com/Graph-COM/CoBFormer](https://github.com/Graph-COM/CoBFormer) Note that the original code is 
  implemented with PyTorch for the paper. 

## Dataset Statics

| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Pubmed   | 19,717  | 88,651  | 3         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

## How to run examples
```bash
TL_BACKEND="torch" python cobformer_trainer.py --dataset Pubmed --hidden_dim 64 --layers 1 --heads 1 --gcn_layers 2 --lr 0.005 --l2_coef 0.001 --drop_rate1 0.5 --drop_rate2 0.1 --alpha 0.7 --tau 0.3 --gpu 1 --n_epoch 200 --seed 42
```

## 	Performance

| Dataset | Metrics | Author's Code (CoB_G) | Author's Code (CoB_T) | GAMMAGL's Code (CoB_G) | GAMMAGL's Code (CoB_T) |
|:-------:|:-------:|:---------------------:|:---------------------:|:----------------------:|:----------------------:|
|  PubMed |  Mi-F1  |         80.52         |         81.42         |         86.20          |         64.30          |
|  PubMed |  Ma-F1  |         80.02         |         81.04         |         85.24          |         62.07          |




