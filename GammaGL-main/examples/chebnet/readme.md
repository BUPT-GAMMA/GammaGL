

# Chebyshev Networks (ChebNet)

- Paper link: [https://arxiv.org/abs/1606.09375](https://arxiv.org/abs/1606.09375)


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
# k=2, available dataset: "cora", "citeseer", "pubmed"
TL_BACKEND="paddle" python chebnet_trainer.py --k 2 --dataset cora --lr 0.01 --hidden_dim 64 --drop_rate 0.8 --l2_coef 0.0005
TL_BACKEND="tensorflow" python chebnet_trainer.py --k 2 --dataset cora --lr 0.01 --hidden_dim 64 --drop_rate 0.7 --l2_coef 0.0005
TL_BACKEND="torch" python chebnet_trainer.py --k 2 --dataset cora --lr 0.01 --hidden_dim 64 --drop_rate 0.7 --l2_coef 0.0005
TL_BACKEND="paddle" python chebnet_trainer.py --k 2 --dataset citeseer --lr 0.01 --hidden_dim 32 --drop_rate 0.9 --l2_coef 0.0005
TL_BACKEND="tensorflow" python chebnet_trainer.py --k 2 --dataset citeseer --lr 0.01 --hidden_dim 64 --drop_rate 0.9 --l2_coef 0.0005
TL_BACKEND="torch" python chebnet_trainer.py --k 2 --dataset citeseer --lr 0.01 --hidden_dim 64 --drop_rate 0.9 --l2_coef 0.0005
TL_BACKEND="paddle" python chebnet_trainer.py --k 2 --dataset pubmed --lr 0.01 --hidden_dim 64 --drop_rate 0.6 --l2_coef 0.0005
TL_BACKEND="tensorflow" python chebnet_trainer.py --k 2 --dataset pubmed --lr 0.01 --hidden_dim 64 --drop_rate 0.6 --l2_coef 0.0005

# k=3, available dataset: "cora", "citeseer"
TL_BACKEND="paddle" python chebnet_trainer.py --k 3 --dataset cora --lr 0.01 --hidden_dim 64 --drop_rate 0.8 --l2_coef 0.0005
TL_BACKEND="tensorflow" python chebnet_trainer.py --k 3 --dataset cora --lr 0.01 --hidden_dim 64 --drop_rate 0.7 --l2_coef 0.0005
TL_BACKEND="torch" python chebnet_trainer.py --k 3 --dataset cora --lr 0.01 --hidden_dim 64 --drop_rate 0.7 --l2_coef 0.0005
TL_BACKEND="paddle" python chebnet_trainer.py --k 3 --dataset citeseer --lr 0.01 --hidden_dim 64 --drop_rate 0.9 --l2_coef 0.0005
TL_BACKEND="torch" python chebnet_trainer.py --k 3 --dataset citeseer --lr 0.01 --hidden_dim 64 --drop_rate 0.9 --l2_coef 0.0005
```

K=2

| Dataset  | Paper | Our(pd)    | Our(tf)    | Our(torch) |
| -------- | ----- | ---------- | ---------- | ---------- |
| cora     | 81.2  | 80.78±1.12 | 80.62±0.80 | 80.42±1.18 |
| citeseer | 69.6  | 69.45±1.75 | 70.34±1.46 | 70.58±0.62 |
| pubmed   | 73.8  | 76.90±1.40 | 75.68±1.42 | OOM        |

K=3

| Dataset  | Paper | Our(pd)    | Our(tf) | Our(torch) |
| -------- | ----- | ---------- | -------  | -------  |
| cora     | 79.5  | 81.88±1.32 |80.55±1.75|81.72±1.38|
| citeseer | 69.8  | 70.27±0.93 | OOM      |70.18±0.82|
| pubmed   | 74.4  | OOM | OOM | OOM |