# Hypergraph Convolution and Hypergraph Attention

- Paper link: https://arxiv.org/pdf/1901.08150.pdf

# Dataset Statics

| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

## Results

```
# available dataset: "cora", "citeseer", "pubmed"
# use attention by adding argument --use_attention=True
TL_BACKEND="paddle" python hcha_trainer.py --dataset cora --lr 0.01 --l2_coef 0.01 --drop_rate 0.9  
TL_BACKEND="paddle" python hcha_trainer.py --dataset citeseer --lr 0.01 --l2_coef 0.01 --drop_rate 0.7 
TL_BACKEND="paddle" python hcha_trainer.py --dataset pubmed --lr 0.01 --l2_coef 0.005 --drop_rate 0.6 
TL_BACKEND="torch" python hcha_trainer.py --dataset cora --lr 0.005 --l2_coef 0.01 --drop_rate 0.8 
TL_BACKEND="torch" python hcha_trainer.py --dataset citeseer --lr 0.01 --l2_coef 0.01 --drop_rate 0.7 
TL_BACKEND="torch" python hcha_trainer.py --dataset pubmed --lr 0.01 --l2_coef 0.002 --drop_rate 0.5 
TL_BACKEND="mindspore" python hcha_trainer.py --dataset cora --lr 0.01 --l2_coef 0.01 --drop_rate 0.6
TL_BACKEND="mindspore" python hcha_trainer.py --dataset citeseer --lr 0.01 --l2_coef 0.05 --drop_rate 0.7 
TL_BACKEND="mindspore" python hcha_trainer.py --dataset pubmed --lr 0.01 --l2_coef 0.01 --drop_rate 0.6 
```

### Hypergraph Convolution

| Dataset  | Paper |  Our(pd)   |  Our(th)   |  Our(ms)   |
| -------- | ----- | ---------- | ---------- | ---------- |
| cora     | 82.19 | 78.58±0.57 | 77.14±1.45 | 78.08±0.21 |
| citeseer | 70.35 | 63.32±1.55 | 63.48±1.39 | 62.98±0.71 |
| pubmed   | 78.4  | 76.42±0.31 | 76.44±0.22 | 76.36±0.25 |

### Hypergraph Attention

| Dataset  | Paper |  Our(pd)   |  Our(th)   |  Our(ms)   |
| -------- | ----- | ---------- | ---------- | ---------- |
| cora     | 82.61 | 77.40±0.93 | 76.84±2.25 |     Err    |
| citeseer | 70.88 | 63.44±1.05 | 62.22±3.02 |     Err    |
| pubmed   | 78.4  | 75.36±0.92 | 74.76±0.42 |     Err    |