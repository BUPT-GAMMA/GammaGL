# Mixture Model Networks (GMM or MoNet)

- Paper link: [https://arxiv.org/abs/1611.08402](https://arxiv.org/abs/1611.08402)

# Dataset Statics

| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Cora     | 2,708   | 10,556  | 7         |
| Pubmed   | 19,717  | 88,651  | 3         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

Results
-------

```bash
# available dataset: "cora", "pubmed"
TL_BACKEND="paddle" python gmm_trainer.py --dataset cora --n_epoch 100 --hidden_dim 32 --num_layers 2 --lr 0.01 --l2_coef 0.01 --drop_rate 0.8
TL_BACKEND="paddle" python gmm_trainer.py --dataset pubmed --n_epoch 200 --hidden_dim 32 --num_layers 2 --lr 0.01 --l2_coef 0.005 --drop_rate 0.9
TL_BACKEND="tensorflow" python gmm_trainer.py --dataset cora --n_epoch 200 --hidden_dim 32 --num_layers 2 --lr 0.01 --l2_coef 0.01 --drop_rate 0.8
TL_BACKEND="tensorflow" python gmm_trainer.py --dataset pubmed --n_epoch 200 --hidden_dim 32 --num_layers 2 --lr 0.01 --l2_coef 0.005 --drop_rate 0.9 
TL_BACKEND="torch" python gmm_trainer.py --dataset cora --n_epoch 200 --hidden_dim 32 --num_layers 2 --lr 0.007 --l2_coef 0.01 --drop_rate 0.8
TL_BACKEND="torch" python gmm_trainer.py --dataset pubmed --n_epoch 200 --hidden_dim 32 --num_layers 2 --lr 0.01 --l2_coef 0.01 --drop_rate 0.9
TL_BACKEND="mindspore" python gmm_trainer.py --dataset cora --n_epoch 100 --hidden_dim 32 --num_layers 2 --lr 0.01 --l2_coef 0.01 --drop_rate 0.8
TL_BACKEND="mindspore" python gmm_trainer.py --dataset pubmed --n_epoch 100 --hidden_dim 16 --num_layers 2 --lr 0.006 --l2_coef 0.005 --drop_rate 0.6
```

| Dataset  | Paper       | Our(pd)     | Our(tf)     | Our(th)     | Our(ms)     |
|----------|-------------|-------------|-------------|-------------|-------------|
| cora     | 81.69±0.48% | 80.90±0.73% | 81.68±0.65% | 81.66±0.45% | 80.06±1.19% |
| pubmed   | 78.81±0.44% | 78.36±0.68% | 79.08±0.53% | 78.84±0.19% | 78.08±0.39% |
