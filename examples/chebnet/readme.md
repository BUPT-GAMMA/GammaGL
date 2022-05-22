# Chebyshev Networks (ChebNet)

- Paper link: [https://arxiv.org/abs/1606.09375](https://arxiv.org/abs/1606.09375)


# Dataset Statics

| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

Results
-------

```bash
# available dataset: "cora", "citeseer", "pubmed"
TL_BACKEND="paddle" python chebnet_trainer.py --k 2 --dataset cora --lr 0.01 --hidden_dim 64 --drop_rate 0.7 --l2_coef 0.0005
TL_BACKEND="tensorflow" python chebnet_trainer.py --k 2 --dataset cora --lr 0.01 --hidden_dim 64 --drop_rate 0.7 --l2_coef 0.0005
TL_BACKEND="paddle" python chebnet_trainer.py --k 2 --dataset citeseer --lr 0.01 --hidden_dim 55 --drop_rate 0.75 --l2_coef 0.005
TL_BACKEND="tensorflow" python chebnet_trainer.py --k 2 --dataset citeseer --lr 0.01 --hidden_dim 55 --drop_rate 0.75 --l2_coef 0.005
```

| Dataset  | Paper | Our(pd)   | Our(tf)    |
|----------|-------|-----------|------------|
| cora     | 81.2  | 80.3±1.1  | 80.1±1.5   |
| citeseer |       | 70.28±0.8 | 67.78±1.22 |
| pubmed   |       | OOM       | OOM        |
GPU:NVIDIA GeForce MX250