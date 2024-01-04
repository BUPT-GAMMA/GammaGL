# AM-GCN: Adaptive Multi-channel Graph Convolutional Networks (AM-GCN)

- Paper link: [https://dl.acm.org/doi/10.1145/3394486.3403177](https://dl.acm.org/doi/10.1145/3394486.3403177)
- Author's code repo: [https://github.com/BUPT-GAMMA/AM-GCN](https://github.com/BUPT-GAMMA/AM-GCN). Note that the original code is 
  implemented with PyTorch for the paper. 

# Dataset Statics

| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

Results
-------

```bash
# available dataset: "cora", "citeseer", "pubmed"
TL_BACKEND=torch python amgcn_trainer.py --dataset cora --lr 0.0005 --k 6 --hidden1 512 --hidden2 32 --drop_rate 0.5 --beta 1e-10 --theta 0.0001
TL_BACKEND=torch python amgcn_trainer.py --dataset citeseer --lr 0.0005 --k 7 --hidden1 768 --hidden2 256 --drop_rate 0.5 --beta 5e-10 --theta 0.001
TL_BACKEND=torch python amgcn_trainer.py --dataset pubmed --lr 0.0005 --k 6 --hidden1 512 --hidden2 128 --drop_rate 0.5 --beta 5e-10 --theta 0.001
```

| Dataset  | Paper | Our(th)    |
| -------- | ----- | ---------- |
| cora     |       | 79.5(±0.3) |
| citeseer | 73.1  | 71.7(±1.2) |
| pubmed   |       | 64.4(±0.8) |