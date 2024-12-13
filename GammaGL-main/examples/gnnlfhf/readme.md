# Interpreting and Unifying Graph Neural Networks with An Optimization Framework（GNNLFHF）

- Paper link: [https://arxiv.org/pdf/2101.11859](https://arxiv.org/pdf/2101.11859)
- Author's code repo: [https://github.com/zhumeiqiBUPT/GNN-LF-HF/tree/main](https://github.com/zhumeiqiBUPT/GNN-LF-HF/tree/main). Note that the original code is implemented with PyTorch for the paper. 

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
TL_BACKEND="torch" python gnnlfhf_trainer.py --dataset cora --model_type GNN-LF --model_form closed --alpha 0.3 --mu 0.1 --beta 0.1 --niter 20 --lr 0.01 --hidden_dim 64 --drop_rate 0.8 --reg_lambda 5e-3
TL_BACKEND="torch" python gnnlfhf_trainer.py --dataset cora --model_type GNN-LF --model_form iterative --alpha 0.3 --mu 0.1 --beta 0.1 --niter 20 --lr 0.01 --hidden_dim 64 --drop_rate 0.8 --reg_lambda 5e-3
TL_BACKEND="torch" python gnnlfhf_trainer.py --dataset cora --model_type GNN-HF --model_form closed --alpha 0.3 --mu 0.1 --beta 0.1 --niter 20 --lr 0.01 --hidden_dim 64 --drop_rate 0.8 --reg_lambda 5e-3
TL_BACKEND="torch" python gnnlfhf_trainer.py --dataset cora --model_type GNN-HF --model_form iterative --alpha 0.3 --mu 0.1 --beta 0.1 --niter 20 --lr 0.01 --hidden_dim 64 --drop_rate 0.8 --reg_lambda 5e-3

TL_BACKEND="torch" python gnnlfhf_trainer.py --dataset citeseer --model_type GNN-LF --model_form closed --alpha 0.3 --mu 0.1 --beta 0.1 --niter 20 --lr 0.01 --hidden_dim 64 --drop_rate 0.8 --reg_lambda 5e-3
TL_BACKEND="torch" python gnnlfhf_trainer.py --dataset citeseer --model_type GNN-LF --model_form iterative --alpha 0.3 --mu 0.1 --beta 0.1 --niter 20 --lr 0.01 --hidden_dim 64 --drop_rate 0.8 --reg_lambda 5e-3
TL_BACKEND="torch" python gnnlfhf_trainer.py --dataset citeseer --model_type GNN-HF --model_form closed --alpha 0.3 --mu 0.1 --beta 0.1 --niter 20 --lr 0.01 --hidden_dim 64 --drop_rate 0.8 --reg_lambda 5e-3
TL_BACKEND="torch" python gnnlfhf_trainer.py --dataset citeseer --model_type GNN-HF --model_form iterative --alpha 0.3 --mu 0.1 --beta 0.1 --niter 20 --lr 0.01 --hidden_dim 64 --drop_rate 0.8 --reg_lambda 5e-3

TL_BACKEND="torch" python gnnlfhf_trainer.py --dataset pubmed --model_type GNN-LF --model_form closed --alpha 0.3 --mu 0.1 --beta 0.1 --niter 20 --lr 0.01 --hidden_dim 64 --drop_rate 0.8 --reg_lambda 5e-3
TL_BACKEND="torch" python gnnlfhf_trainer.py --dataset pubmed --model_type GNN-LF --model_form iterative --alpha 0.3 --mu 0.1 --beta 0.1 --niter 20 --lr 0.01 --hidden_dim 64 --drop_rate 0.8 --reg_lambda 5e-3
TL_BACKEND="torch" python gnnlfhf_trainer.py --dataset pubmed --model_type GNN-HF --model_form closed --alpha 0.3 --mu 0.1 --beta 0.1 --niter 20 --lr 0.01 --hidden_dim 64 --drop_rate 0.8 --reg_lambda 5e-3
TL_BACKEND="torch" python gnnlfhf_trainer.py --dataset pubmed --model_type GNN-HF --model_form iterative --alpha 0.3 --mu 0.1 --beta 0.1 --niter 20 --lr 0.01 --hidden_dim 64 --drop_rate 0.8 --reg_lambda 5e-3
```

| Dataset  | Model         | Paper      | Our(th)    |
| -------- | ------------- | ---------- | ---------- |
| cora     | GNN-LF-closed | 83.70±0.14 | 82.05±0.98 |
| cora     | GNN-LF-iter   | 83.53±0.24 | 81.81±0.65 |
| cora     | GNN-HF-closed | 83.96±0.22 | 82.48±1.18 |
| cora     | GNN-HF-iter   | 83.79±0.29 | 81.28±0.69 |
| citeseer | GNN-LF-closed | 71.98±0.33 | 70.51±1.08 |
| citeseer | GNN-LF-iter   | 71.92±0.24 | 71.11±1.38 |
| citeseer | GNN-HF-closed | 72.30±0.28 | 70.24±1.01 |
| citeseer | GNN-HF-iter   | 72.03±0.36 | 70.14±1.52 |
| pubmed   | GNN-LF-closed | 80.34±0.18 | 75.14±0.89 |
| pubmed   | GNN-LF-iter   | 80.33±0.20 | 76.68±0.58 |
| pubmed   | GNN-HF-closed | 80.41±0.25 | 76.36±0.71 |
| pubmed   | GNN-HF-iter   | 80.54±0.25 | 78.02±0.28 |
