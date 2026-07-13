# GENeralized Aggregation Networks(GEN)

- Paper link: [https://arxiv.org/abs/2006.07739](https://arxiv.org/abs/2006.07739)
- Author's code (No official code found)
- Community Code: [https://paperswithcode.com/paper/deepergcn-all-you-need-to-train-deeper-gcns](https://paperswithcode.com/paper/deepergcn-all-you-need-to-train-deeper-gcns)

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
TL_BACKEND="torch" python gen_trainer.py --dataset cora --num_layers 2 --lr 0.01 --l2_coef 0.01 --drop_rate 0.3 --agg mean
TL_BACKEND="torch" python gen_trainer.py --dataset cora --num_layers 2 --lr 0.01 --l2_coef 0.0002 --drop_rate 0.3 --agg sum
TL_BACKEND="torch" python gen_trainer.py --dataset cora --num_layers 5 --lr 0.01 --l2_coef 0.0002 --drop_rate 0.5 --agg sum
TL_BACKEND="torch" python gen_trainer.py --dataset cora --num_layers 5 --lr 0.01 --l2_coef 0.0002 --drop_rate 0.5 --agg mean
TL_BACKEND="torch" python gen_trainer.py --dataset cora --num_layers 2 --lr 0.01 --l2_coef 0.0001 --drop_rate 0.4 --powermean 
TL_BACKEND="torch" python gen_trainer.py --dataset pubmed --num_layers 2 --lr 0.01 --l2_coef 0.01 --drop_rate 0.6 
```

## 
| aggregation  |    sum     |    mean       | softmax    | powermean  |
|----------    |  -------   | ------------  |------------|------------|
| PlainGCN     | 77.90±0.60 |  78.40±0.90   | ——————     | ——————     |
| ResGCN       | 76.30±0.80 |  75.80±0.70   | ——————     | ——————     |
| ResGEN       |   ——————   |    ——————     | 78.28±1.08 | 77.80±1.50 |
| DyResGEN     |   ——————   |    ——————     | 75.65±1.25 | 76.90±1.20 |
