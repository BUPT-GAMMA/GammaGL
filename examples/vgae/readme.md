# Variational Graph Auto-Encoders (VGAE)

- Paper link: [https://arxiv.org/abs/1611.07308](https://arxiv.org/abs/1611.07308)
- Author's code repo: [https://github.com/tkipf/gae](https://github.com/tkipf/gae). Note that the original code is 
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
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset cora --model VGAE --features 1 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset citeseer --model VGAE --features 1 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset pubmed --model VGAE --features 1 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset cora --model VGAE --features 1 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset citeseer --model VGAE --features 1 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset pubmed --model VGAE --features 1 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset cora --model GAE --features 1 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset citeseer --model GAE --features 1 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset pubmed --model GAE --features 1 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset cora --model GAE --features 1 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset citeseer --model GAE --features 1 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset pubmed --model GAE --features 1 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset cora --model VGAE --features 0 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset citeseer --model VGAE --features 0 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset pubmed --model VGAE --features 0 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset cora --model VGAE --features 0 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset citeseer --model VGAE --features 0 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset pubmed --model VGAE --features 0 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset cora --model GAE --features 0 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset citeseer --model GAE --features 0 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset pubmed --model GAE --features 0 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset cora --model GAE --features 0 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset citeseer --model GAE --features 0 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset pubmed --model GAE --features 0 --n_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
```

VGAE* and GAE* denote experiments without using input features, GAE and VGAE use input features. 
We report area under the ROC curve (AUC) and average precision (AP) scores for each model on the test set.

| Dataset  | Paper(VGAE)(AUC,AP) | Our(tf)(VGAE)(AUC,AP) | Our(th)(VGAE)(AUC,AP) |
|----------|:-------------------:|:---------------------:|:---------------------:|
| cora     |     91.4   92.6     | 92.91±0.62 93.99±0.87 | 90.80±0.32 91.51±0.74 |
| citeseer |     90.8   92.0     | 91.48±0.56 93.11±0.12 | 90.81±0.34 91.99±0.47 |
| pubmed   |     94.4   94.7     | 93.91±0.72 93.79±0.65 | 94.45±0.24 94.86±0.35 |


| Dataset  | Paper(VGAE*)(AUC,AP) | Our(tf)(VGAE)(AUC,AP) | Our(th)(VGAE)(AUC,AP) |
|----------|:--------------------:|:---------------------:|:---------------------:|
| cora     |     84.0   87.7      | 84.35±0.21 88.11±0.68 | 83.42±0.82 88.05±0.27 |
| citeseer |     78.9   84.1      | 79.27±0.36 83.36±0.52 | 79.91±0.26 84.33±0.27 |
| pubmed   |     82.7   87.5      | 82.97±0.51 86.95±0.86 | 81.97±0.78 86.96±0.15 |


| Dataset  | Paper(GAE)(AUC,AP) | Our(tf)(VGAE)(AUC,AP) | Our(th)(VGAE)(AUC,AP) |
|----------|:------------------:|:---------------------:|:---------------------:|
| cora     |    91.0   92.0     | 91.30±0.85 92.42±0.43 | 92.02±0.44 93.12±0.16 |
| citeseer |    89.5   89.9     | 87.06±0.14 88.18±0.26 | 89.62±0.48 89.86±0.73 |
| pubmed   |    96.4   96.5     | 97.06±0.32 96.68±0.31 | 97.11±0.56 97.13±0.23 |


| Dataset  | Paper(GAE*)(AUC,AP) | Our(tf)(VGAE)(AUC,AP) | Our(th)(VGAE)(AUC,AP) |
|----------|:-------------------:|:---------------------:|:---------------------:|
| cora     |     84.3   88.1     | 85.88±0.22 89.55±0.77 | 83.78±0.71 87.28±0.88 |
| citeseer |     78.7   84.1     | 77.45±0.66 83.76±0.32 | 78.23±0.19 85.21±0.47 |
| pubmed   |     82.2   87.4     | 83.02±0.13 87.32±0.55 | 83.53±0.29 87.95±0.66 |
