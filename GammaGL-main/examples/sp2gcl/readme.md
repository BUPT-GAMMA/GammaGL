# Graph Contrastive Learning with Stable and Scalable

- Paper link: [https://proceedings.neurips.cc/paper_files/paper/2023/file/8e9a6582caa59fda0302349702965171-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/8e9a6582caa59fda0302349702965171-Paper-Conference.pdf)
- Author's code repo: [https://github.com/bdy9527/Sp2GCL](https://github.com/bdy9527/Sp2GCL).

# Dataset Statics

| Dataset  | # Nodes | # Edges  | # Classes |
|----------|---------|----------|-----------|
| PubMed   | 19,717  | 88,648   | 3         |
| Wiki-CS  | 11,701  | 216,123  | 10        |
| Facebook | 22,470  | 342,004  | 4         |



Results
-------

```bash
TL_BACKEND="torch" python sp2gcl_trainer.py --dataset pubmed --hidden_dim 4096 --spe_dim 30 --output_dim 32 --lr 0.001 --period 50 --l2_coef 5e-4 --n_epoch 5
TL_BACKEND="torch" python sp2gcl_trainer.py --dataset wikics --hidden_dim 2048 --spe_dim 100 --output_dim 32 --lr 0.001 --period 10 --l2_coef 5e-1 --n_epoch 3
TL_BACKEND="torch" python sp2gcl_trainer.py --dataset facebook --hidden_dim 1500 --spe_dim 100 --output_dim 32 --lr 0.001 --period 10 --l2_coef 5e-4 --n_epoch 5
```


# Dataset Statics

| Dataset  | Paper Code | Out(th)    |
| -------- | ---------- | ---------- |
| PubMed   | 82.3±0.3   | 78.66±0.76 |
| Wiki-CS  | 79.42±0.19 | 78.64±0.20 |
| Facebook | 90.43±0.13 | 87.53±0.34 |

