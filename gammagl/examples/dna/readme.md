# JUST JUMP: DYNAMIC NEIGHBORHOOD AGGREGATION IN GRAPH NEURAL NETWORKS (DNA)

- Paper link: [https://arxiv.org/abs/1904.04849](https://arxiv.org/abs/1904.04849)
- The implementation of PyG: [https://github.com/pyg-team/pytorch_geometric/blob/master/examples/dna.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/dna.py). 

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

TL_BACKEND="torch" python dna_trainer.py --dataset cora --lr 0.01 --drop_rate_conv 0.2 --drop_rate_model 0.8 --num_layers 3 --heads 64 --groups 1 --l2_coef 5e-5 --hidden_dim 256
TL_BACKEND="torch" python dna_trainer.py --dataset cora --lr 0.01 --drop_rate_conv 0.2 --drop_rate_model 0.9 --num_layers 3 --heads 32 --groups 8 --l2_coef 5e-5 --hidden_dim 256
TL_BACKEND="torch" python dna_trainer.py --dataset cora --lr 0.01 --drop_rate_conv 0.1 --drop_rate_model 0.8 --num_layers 4 --heads 8 --groups 16 --l2_coef 5e-5 --hidden_dim 128
TL_BACKEND="torch" python dna_trainer.py --dataset citeseer --lr 0.01 --drop_rate_conv 0.2 --drop_rate_model 0.8 --num_layers 3 --heads 32 --groups 1 --l2_coef 5e-5 --hidden_dim 128
TL_BACKEND="torch" python dna_trainer.py --dataset citeseer --lr 0.01 --drop_rate_conv 0.1 --drop_rate_model 0.8 --num_layers 4 --heads 8 --groups 8 --l2_coef 5e-5 --hidden_dim 128
TL_BACKEND="torch" python dna_trainer.py --dataset citeseer --lr 0.01 --drop_rate_conv 0.1 --drop_rate_model 0.8 --num_layers 4 --heads 8 --groups 16 --l2_coef 5e-5 --hidden_dim 128
TL_BACKEND="torch" python dna_trainer.py --dataset pubmed --lr 0.01 --drop_rate_conv 0.1 --drop_rate_model 0.8 --num_layers 4 --heads 8 --groups 1 --l2_coef 5e-5 --hidden_dim 128
TL_BACKEND="torch" python dna_trainer.py --dataset pubmed --lr 0.01 --drop_rate_conv 0.1 --drop_rate_model 0.8 --num_layers 4 --heads 8 --groups 8 --l2_coef 5e-5 --hidden_dim 128
TL_BACKEND="torch" python dna_trainer.py --dataset pubmed --lr 0.01 --drop_rate_conv 0.1 --drop_rate_model 0.8 --num_layers 4 --heads 8 --groups 16 --l2_coef 5e-5 --hidden_dim 128
```

| Dataset  | group | Paper      | Our(th)    |
| -------- | ----- | ---------- | ---------- |
| cora     | 1     | 83.88±0.50 | 80.50±0.81 |
| cora     | 8     | 85.86±0.45 | 81.22±0.18 |
| cora     | 16    | 86.15±0.57 | 82.25±0.4  |
| citeseer | 1     | 73.37±0.83 | 71.41±1.02 |
| citeseer | 8     | 74.19±0.66 | 72.29±0.59 |
| citeseer | 16    | 74.50±0.62 | 72.99±0.68 |
| pubmed   | 1     | 87.80±0.25 | 87.32±0.52 |
| pubmed   | 8     | 88.04±0.17 | 87.46±0.25 |
| pubmed   | 16    | 88.04±0.22 | 87.49±0.11 |

![image-20240703225836573](readme.assets/image-20240703225836573.png)
