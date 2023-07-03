# Confidence Calibration Graph Convolutional Networks (CAGCN)

- Paper link: [https://arxiv.org/abs/2109.14285](https://arxiv.org/abs/2109.14285)
- Author's code repo: [https://github.com/BUPT-GAMMA/CaGCN](https://github.com/BUPT-GAMMA/CaGCN). Note that the original code is 
  implemented with Torch for the paper. 

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
# base model: GCN (only)

## GCN L/C=20

TL_BACKEND="tensorflow" python cagcn_trainer.py --hidden 64 --dataset Cora --labelrate 20 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8
TL_BACKEND="tensorflow" python cagcn_trainer.py --hidden 64 --dataset Citeseer --labelrate 20 --stage 5 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.9
TL_BACKEND="tensorflow" python cagcn_trainer.py --model GCN --hidden 64 --dataset Pubmed --labelrate 20 --stage 6 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8
TL_BACKEND="torch" python cagcn_trainer.py --hidden 64 --dataset Cora --labelrate 20 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8
TL_BACKEND="torch" python cagcn_trainer.py --hidden 64 --dataset Citeseer --labelrate 20 --stage 5 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.9
TL_BACKEND="torch" python cagcn_trainer.py --model GCN --hidden 64 --dataset Pubmed --labelrate 20 --stage 6 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8


## GCN L/C=40

TL_BACKEND="tensorflow" python cagcn_trainer.py --hidden 64 --dataset Cora --labelrate 40 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8
TL_BACKEND="tensorflow" python cagcn_trainer.py --hidden 64 --dataset Citeseer --labelrate 40 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.85
TL_BACKEND="tensorflow" python cagcn_trainer.py --hidden 64 --dataset Pubmed --labelrate 40 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8
TL_BACKEND="torch" python cagcn_trainer.py --hidden 64 --dataset Cora --labelrate 20 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8
TL_BACKEND="torch" python cagcn_trainer.py --hidden 64 --dataset Citeseer --labelrate 20 --stage 5 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.9
TL_BACKEND="torch" python cagcn_trainer.py --model GCN --hidden 64 --dataset Pubmed --labelrate 20 --stage 6 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8


## GCN L/C=60

TL_BACKEND="tensorflow" python cagcn_trainer.py --hidden 64 --dataset Cora --labelrate 60 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8
TL_BACKEND="tensorflow" python cagcn_trainer.py --hidden 64 --dataset Citeseer --labelrate 60 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.8
TL_BACKEND="tensorflow" python cagcn_trainer.py --hidden 64 --dataset Pubmed --labelrate 60 --stage 3 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.6
TL_BACKEND="torch" python cagcn_trainer.py --hidden 64 --dataset Cora --labelrate 20 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8
TL_BACKEND="torch" python cagcn_trainer.py --hidden 64 --dataset Citeseer --labelrate 20 --stage 5 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.9
TL_BACKEND="torch" python cagcn_trainer.py --model GCN --hidden 64 --dataset Pubmed --labelrate 20 --stage 6 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8


```

| Dataset  | L/C |   Paper    |  Our(tf)   |  Our(th)   |
|----------|-----|------------|------------|------------|
|   cora   | 20  | 83.11±0.52 |  | 82.67±0.36 |
|   cora   | 40  | 84.37±0.38 |  |  |
|   cora   | 60  | 85.79±0.27 |  |  |
| citeseer | 20  | 74.90±0.40 |  |  |
| citeseer | 40  | 75.48±0.50 |  |  |
| citeseer | 60  | 76.43±0.20 |  |  |
|  pubmed  | 20  | 81.16±0.36 |  |  |
|  pubmed  | 40  | 83.08±0.21 |  |  |
|  pubmed  | 60  | 84.47±0.23 |  |  |

cora 20: torch 83.00 82.30 83.00 82.10 82.3 83.2 82.6 82.9 82.9 82.4
         tensorflow: 82.4 81.6 