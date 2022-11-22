# Graph Neural Networks with Feature-wise Linear Modulation (GNN-FiLM)

- Paper link: [https://arxiv.org/abs/1906.12192](https://arxiv.org/abs/1906.12192)
- Author's code repo: [https://github.com/ Microsoft/tf-gnn-samples](https://github.com/ Microsoft/tf-gnn-samples)

# Dataset Statics

| Dataset | #Graphs | # Nodes | # Edges | # Classes |
|---------|---------|---------|---------|-----------|
| PPI     | 20      | 44906   | 1226368 | 121       |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

Results
-------

```bash
TL_BACKEND="tensorflow" python film_trainer.py --dataset ppi --lr 0.001 --l2_coef 5e-4 --drop_rate 0.1 --hidden_dim 160  --num_layers 4
TL_BACKEND="torch" python film_trainer.py --dataset ppi --lr 0.001 --l2_coef 5e-4 --drop_rate 0.1 --hidden_dim 160 --num_layers 4
TL_BACKEND="paddle" python film_trainer.py --dataset ppi --lr 0.001 --l2_coef 5e-4 --drop_rate 0.1 --hidden_dim 160 --num_layers 4
```

| Dataset | Paper    | Our(tf) | Our(torch) | Our(pd) |
|---------|----------|---------|------------|--------|
| PPI     | 84.1±0.5 | 70±0.7  | 94±0.7     | 70±0.7 |

