# Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning

- Paper link: [https://arxiv.org/abs/2105.09111](https://arxiv.org/abs/2105.09111)
- Author's code repo: [https://github.com/liun-online/HeCo](https://github.com/liun-online/HeCo)

Dataset Statics
-------
| Dataset  | # Nodes_paper | # Nodes_author | # Nodes_subject |
|----------|---------------|----------------|-----------------|
| ACM      | 4019          | 7167           | 60              |

Refer to [ACM](https://github.com/AndyJZhao/NSHE/tree/master/data/acm).

Results For ACM
-------
```bash
TL_BACKEND="torch" python HeCo_trainer.py --dataset acm --hidden_dim 64  --nb_epochs 10000 --eva_lr 0.05 --lr 0.0075 --l2_coef 0 --tau 0.8 --lam 0.5 --feat_drop 0.3 --attn_drop 0.3
TL_BACKEND="paddle" python HeCo_trainer.py --dataset acm --hidden_dim 64  --nb_epochs 10000 --eva_lr 0.05 --lr 0.0075 --l2_coef 0 --tau 0.8 --lam 0.5 --feat_drop 0.3 --attn_drop 0.3
TL_BACKEND="tensorflow" python HeCo_trainer.py --dataset acm --hidden_dim 64  --nb_epochs 10000 --eva_lr 0.05 --lr 0.0075 --l2_coef 0 --tau 0.8 --lam 0.5 --feat_drop 0.3 --attn_drop 0.3
```
- Ma-F1

| number of train_labels  | Paper    | Our(tf)  | Our(pd)  | Our(torch) |
|-------------------------|----------|----------|----------|------------|
|    20                   | 88.56±0.8| 84.7±0.4 | 85.0±0.4 | 85.0±0.3   |
|    40                   | 87.61±0.5| 88.1±0.3 | 88.63±0.1| 88.64±0.2  |
|    60                   | 89.04±0.5| 87.4±0.4 | 88.3±0.4 | 88.4±0.6   |

- Mi-F1

| number of train_labels  | Paper    | Our(tf)  | Our(pd)  | Our(torch) |
|-------------------------|----------|----------|----------|------------|
|    20                   | 88.13±0.8| 84.1±0.4 | 84.8±0.8 | 85.0±0.4   |
|    40                   | 87.45±0.5| 87.9±0.3 | 88.43±0.1| 88.53±0.6  |
|    60                   | 88.71±0.5| 87.4±0.4 | 88.2±0.5 | 88.45±0.6  |


- AUC

| number of train_labels  | Paper    | Our(tf)  | Our(pd)  | Our(torch) |
|-------------------------|----------|----------|----------|------------|
|    20                   | 96.49±0.3| 93.8±0.4 | 95.1±0.4 | 95.3±0.3   |
|    40                   | 96.4±0.4 | 96.4±0.3 | 97.1±0.2 | 97.4±0.3   |
|    60                   | 96.55±0.3| 95.8±0.4 | 96.4±0.4 | 96.7±0.4   |

For TensorFlow runs more slowly than paddlepaddle and pytorch, thus pd and torch are more recommended.
