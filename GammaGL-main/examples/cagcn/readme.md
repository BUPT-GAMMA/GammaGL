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
## GCN L/C=20
TL_BACKEND="torch" python cagcn_trainer.py --hidden 64 --dataset Cora --labelrate 20 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8 --lr 0.005 --weight_decay 0.01 --dropout 0.8
TL_BACKEND="torch" python cagcn_trainer.py --model GCN --dataset Citeseer --stage 7 --epochs 2000 --epoch_for_st 200 --lr 0.01 --lr_for_cal 0.01 --weight_decay 0.0005 --l2_for_cal 0.005 --hidden 256 --dropout 0.8 --labelrate 20 --n_bins 20 --Lambda 0.5 --patience 100 --threshold 0.95
TL_BACKEND="torch" python cagcn_trainer.py --model GCN --hidden 64 --dataset Pubmed --labelrate 20 --stage 6 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8 --lr 0.01 --weight_decay 0.002 --dropout 0.5
TL_BACKEND="tensorflow" python cagcn_trainer.py --hidden 64 --dataset Cora --labelrate 20 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8 --lr 0.005 --weight_decay 0.01 --dropout 0.8 
TL_BACKEND="tensorflow" python cagcn_trainer.py --hidden 64 --dataset Citeseer --labelrate 20 --stage 5 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.9 --lr 0.01 --weight_decay 0.01 --dropout 0.8 
TL_BACKEND="tensorflow" python cagcn_trainer.py --model GCN --hidden 64 --dataset Pubmed --labelrate 20 --stage 6 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8 --lr 0.01 --weight_decay 0.002 --dropout 0.5
TL_BACKEND="paddle" python cagcn_trainer.py --hidden 64 --dataset Cora --labelrate 20 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8 --lr 0.005 --weight_decay 0.01 --dropout 0.8
TL_BACKEND="paddle" python cagcn_trainer.py --model GCN --dataset Citeseer --stage 7 --epochs 2000 --epoch_for_st 200 --lr 0.01 --lr_for_cal 0.01 --weight_decay 0.0005 --l2_for_cal 0.005 --hidden 256 --dropout 0.8 --labelrate 20 --n_bins 20 --Lambda 0.5 --patience 100 --threshold 0.95
TL_BACKEND="paddle" python cagcn_trainer.py --model GCN --hidden 64 --dataset Pubmed --labelrate 20 --stage 6 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8 --lr 0.01 --weight_decay 0.002 --dropout 0.5

## GCN L/C=40
TL_BACKEND="torch" python cagcn_trainer.py --model GCN --hidden 64 --dataset Cora --labelrate 40 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-4 --epoch_for_st 200 --threshold 0.8 --lr 0.005 --weight_decay 0.01 --dropout 0.8
TL_BACKEND="torch" python cagcn_trainer.py --model GCN --dataset Citeseer --stage 4 --epochs 2000 --epoch_for_st 200 --lr 0.05 --lr_for_cal 0.05 --weight_decay 0.05 --l2_for_cal 0.005 --hidden 256 --dropout 0.5 --labelrate 40 --n_bins 20 --Lambda 0.5 --patience 100 --threshold 0.9 
TL_BACKEND="torch" python cagcn_trainer.py --model GCN --hidden 64 --dataset Pubmed --labelrate 40 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8 --lr 0.01 --weight_decay 0.002 --dropout 0.5 
TL_BACKEND="tensorflow" python cagcn_trainer.py --model GCN --hidden 64 --dataset Cora --labelrate 40 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8  --lr 0.005 --weight_decay 0.01 --dropout 0.8 
TL_BACKEND="tensorflow" python cagcn_trainer.py --hidden 64 --dataset Citeseer --labelrate 40 --stage 5 --lr_for_cal 0.001 --l2_for_cal 5e-4 --epoch_for_st 150 --threshold 0.95 --lr 0.01 --weight_decay 0.01 --dropout 0.8
TL_BACKEND="tensorflow" python cagcn_trainer.py --model GCN --hidden 64 --dataset Pubmed --labelrate 40 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8  --lr 0.01 --weight_decay 0.001 --dropout 0.9
TL_BACKEND="paddle" python cagcn_trainer.py --model GCN --hidden 64 --dataset Cora --labelrate 40 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-4 --epoch_for_st 200 --threshold 0.8 --lr 0.005 --weight_decay 0.01 --dropout 0.8
TL_BACKEND="paddle" python cagcn_trainer.py --model GCN --dataset Citeseer --stage 4 --epochs 2000 --epoch_for_st 200 --lr 0.05 --lr_for_cal 0.05 --weight_decay 0.05 --l2_for_cal 0.005 --hidden 256 --dropout 0.5 --labelrate 40 --n_bins 20 --Lambda 0.5 --patience 100 --threshold 0.9
TL_BACKEND="paddle" python cagcn_trainer.py --model GCN --dataset Pubmed --stage 4 --epochs 2000 --epoch_for_st 100 --lr 0.01 --lr_for_cal 0.001 --weight_decay 0.001 --l2_for_cal 0.005 --hidden 64 --dropout 0.9 --labelrate 40 --n_bins 20 --Lambda 0.5 --patience 100 --threshold 0.8


## GCN L/C=60
TL_BACKEND="torch" python cagcn_trainer.py --model GCN --hidden 64 --dataset Cora --labelrate 60 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8 --lr 0.005 --weight_decay 0.01 --dropout 0.8
TL_BACKEND="torch" python cagcn_trainer.py --model GCN --hidden 64 --dataset Citeseer --labelrate 60 --stage 6 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.99 --lr 0.01 --weight_decay 0.01 --dropout 0.7
TL_BACKEND="torch" python cagcn_trainer.py --model GCN --hidden 64 --dataset Pubmed --labelrate 60 --stage 3 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.6 --lr 0.01 --weight_decay 0.002 --dropout 0.5
TL_BACKEND="tensorflow" python cagcn_trainer.py --model GCN --hidden 64 --dataset Cora --labelrate 60 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8 --lr 0.005 --weight_decay 0.01 --dropout 0.8
TL_BACKEND="tensorflow" python cagcn_trainer.py --model GCN --hidden 64 --dataset Citeseer --labelrate 60 --stage 6 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.99 --lr 0.01 --weight_decay 0.01 --dropout 0.8
TL_BACKEND="tensorflow" python cagcn_trainer.py --model GCN --hidden 64 --dataset Pubmed --labelrate 60 --stage 3 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.6  --lr 0.01 --weight_decay 0.001 --dropout 0.9
TL_BACKEND="paddle" python cagcn_trainer.py --model GCN --hidden 64 --dataset Cora --labelrate 60 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8 --lr 0.005 --weight_decay 0.01 --dropout 0.8
TL_BACKEND="paddle" python cagcn_trainer.py --model GCN --hidden 64 --dataset Citeseer --labelrate 60 --stage 6 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.99 --lr 0.01 --weight_decay 0.01 --dropout 0.7
TL_BACKEND="paddle" python cagcn_trainer.py --model GCN --dataset Pubmed --stage 3 --epochs 2000 --epoch_for_st 100 --lr 0.01 --lr_for_cal 0.001 --weight_decay 0.001 --l2_for_cal 0.005 --hidden 64 --dropout 0.9 --labelrate 60 --n_bins 20 --Lambda 0.5 --patience 100 --threshold 0.6
```

| Dataset  | L/C  | Paper      | Our(torch)   | Our(tensorflow) | Our(paddle)  |
| -------- | ---- | ---------- | ------------ | --------------- | ------------ |
| cora     | 20   | 83.11±0.52 | 83.23(±0.43) | 83.39(±0.45)    | 79.86(±1.14) |
| cora     | 40   | 84.37±0.38 | 83.53(±0.63) | 83.73(±0.75)    | 84.42(±0.37) |
| cora     | 60   | 85.79±0.27 | 85.23(±0.56) | 84.07(±1.10)    | 85.22(±0.62) |
| citeseer | 20   | 74.90±0.40 | 72.03(±0.68) | 83.63(±0.34)    | 71.34(±0.91) |
| citeseer | 40   | 75.48±0.50 | 72.63(±0.33) | 84.08(±0.62)    | 72.04(±0.48) |
| citeseer | 60   | 76.43±0.20 | 72.65(±0.68) | 84.86(±0.55)    | 72.46(±1.07) |
| pubmed   | 20   | 81.16±0.36 | 78.13(±0.46) | 82.60(±0.64)    | 78.24(±0.76) |
| pubmed   | 40   | 83.08±0.21 | 78.67(±1.14) | 83.46(±1.23)    | 78.81(±0.99) |
| pubmed   | 60   | 84.47±0.23 | 79.67(±1.52) | 85.28(±0.41)    | 79.96(±0.54) |
