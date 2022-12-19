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

# Model

## GAE

Results
-------
GAE* denotes experiments without using input features, GAE and VGAE use input features.
We report area under the ROC curve (AUC) and average precision (AP) scores for each model on the test set.

```bash
# available dataset: "cora", "citeseer", "pubmed"
# GAE model with input features
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset cora --model GAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset citeseer --model GAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset pubmed --model GAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset cora --model GAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset citeseer --model GAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset pubmed --model GAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="paddle" python vgae_trainer.py --dataset cora --model GAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="paddle" python vgae_trainer.py --dataset citeseer --model GAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="paddle" python vgae_trainer.py --dataset pubmed --model GAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
```
| Dataset  | Paper(GAE)(AUC,AP) | Our(tf)(GAE)(AUC,AP)  | Our(th)(GAE)(AUC,AP)  | Our(pd)(GAE)(AUC,AP)  |
|----------|:------------------:|:---------------------:|:---------------------:|:---------------------:|
| cora     |    91.0   92.0     | 91.30±0.85 92.42±0.43 | 92.02±0.44 93.12±0.16 | 91.16±0.73 92.04±0.87 |
| citeseer |    89.5   89.9     | 87.06±0.14 88.18±0.26 | 89.62±0.48 89.86±0.73 | 89.61±1.34 90.09±1.56 |
| pubmed   |    96.4   96.5     | 97.06±0.32 96.68±0.31 | 97.11±0.56 97.13±0.23 | 96.25±0.29 96.35±0.34 |

```bash
# available dataset: "cora", "citeseer", "pubmed"
# GAE model without input features
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset cora --model GAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset citeseer --model GAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset pubmed --model GAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset cora --model GAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset citeseer --model GAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset pubmed --model GAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="paddle" python vgae_trainer.py --dataset cora --model GAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="paddle" python vgae_trainer.py --dataset citeseer --model GAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="paddle" python vgae_trainer.py --dataset pubmed --model GAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
```

| Dataset  | Paper(GAE*)(AUC,AP) | Our(tf)(GAE)(AUC,AP)  | Our(th)(GAE)(AUC,AP)  | Our(pd)(GAE)(AUC,AP)  |
|----------|:-------------------:|:---------------------:|:---------------------:|:---------------------:|
| cora     |     84.3   88.1     | 85.88±0.22 89.55±0.77 | 83.78±0.71 87.28±0.88 | 85.56±1.41 89.28±1.15 |
| citeseer |     78.7   84.1     | 77.45±0.66 83.76±0.32 | 78.23±0.19 85.21±0.47 | 78.91±1.40 83.93±0.65 |
| pubmed   |     82.2   87.4     | 83.02±0.13 87.32±0.55 | 83.53±0.29 87.95±0.66 | 80.62±0.68 86.58±0.47 |

## VGAE

Results
-------
VGAE* denotes experiments without using input features, GAE and VGAE use input features. 
We report area under the ROC curve (AUC) and average precision (AP) scores for each model on the test set.

```bash
# available dataset: "cora", "citeseer", "pubmed"
# VGAE model with input features
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset cora --model VGAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset citeseer --model VGAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset pubmed --model VGAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset cora --model VGAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset citeseer --model VGAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset pubmed --model VGAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="paddle" python vgae_trainer.py --dataset cora --model VGAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="paddle" python vgae_trainer.py --dataset citeseer --model VGAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="paddle" python vgae_trainer.py --dataset pubmed --model VGAE --features 1 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
```

| Dataset  | Paper(VGAE)(AUC,AP) | Our(tf)(VGAE)(AUC,AP) | Our(th)(VGAE)(AUC,AP) | Our(pd)(VGAE)(AUC,AP) |
|----------|:-------------------:|:---------------------:|:---------------------:|:---------------------:|
| cora     |     91.4   92.6     | 92.91±0.62 93.99±0.87 | 90.80±0.32 91.51±0.74 | 91.42±0.23 92.56±0.54 |
| citeseer |     90.8   92.0     | 91.48±0.56 93.11±0.12 | 90.81±0.34 91.99±0.47 | 90.39±1.27 91.32±1.49 |
| pubmed   |     94.4   94.7     | 93.91±0.72 93.79±0.65 | 94.45±0.24 94.86±0.35 | 95.41±0.16 95.48±0.20 |

```bash
# available dataset: "cora", "citeseer", "pubmed"
# VGAE model without input features
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset cora --model VGAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset citeseer --model VGAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="tensorflow" python vgae_trainer.py --dataset pubmed --model VGAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset cora --model VGAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset citeseer --model VGAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="torch" python vgae_trainer.py --dataset pubmed --model VGAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="paddle" python vgae_trainer.py --dataset cora --model VGAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="paddle" python vgae_trainer.py --dataset citeseer --model VGAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
TL_BACKEND="paddle" python vgae_trainer.py --dataset pubmed --model VGAE --features 0 --num_layers 2 --lr 0.01 --l2_coef 0. --drop_rate 0.
```

| Dataset  | Paper(VGAE*)(AUC,AP) | Our(tf)(VGAE)(AUC,AP) | Our(th)(VGAE)(AUC,AP) | Our(pd)(VGAE)(AUC,AP) |
|----------|:--------------------:|:---------------------:|:---------------------:|:---------------------:|
| cora     |     84.0   87.7      | 84.35±0.21 88.11±0.68 | 83.42±0.82 88.05±0.27 | 84.76±0.76 88.04±0.70 |
| citeseer |     78.9   84.1      | 79.27±0.36 83.36±0.52 | 79.91±0.26 84.33±0.27 | 77.13±0.91 81.84±0.63 |
| pubmed   |     82.7   87.5      | 82.97±0.51 86.95±0.86 | 81.97±0.78 86.96±0.15 | 84.53±3.74 86.60±0.55 |



