# Graph Group Discrimination (GGD)

- Paper link: [https://arxiv.org/abs/2206.01535](https://arxiv.org/abs/2206.01535)
- Author's code repo: [https://github.com/tkipf/gcn](https://github.com/zyzisastudyreallyhardguy/Graph-Group-Discrimination). Note that the original code is 
  implemented with Pytorch for the paper. 

# Dataset Statics

| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |
| Computers| 13,752  | 491,722 | 10        |
| Photo    | 7,650   | 238,162 | 8         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid) and [Amazon](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Amazon).

Cora: python ggd_trainer.py
Citeseer: python ggd_trainer.py --dataset citeseer
Pubmed: python ggd_trainer.py --dataset pubmed
Computers: python ggd_trainer.py --classifier_epochs 3500 --np_epochs 1500 --lr 0.0001 --dataset computers
Photo: python ggd_trainer.py --classifier_epochs 2000 --np_epochs 2000 --lr 0.0005 --dataset photo

| Dataset  | Paper     | Our(pd)    | Our(tf)    | Our(th)    | Our(ms)    |
|----------|-----------|------------|------------|------------|------------|
| cora     | 83.9±0.4  | | | 81.4±0.7 | |
| citeseer | 73.0±0.6  | | | 81.1±0.7 | |
| pubmed   | 81.3±0.8  | | | 81.4±0.2 | |
| computers| 90.1±0.9  | | | 80.8±0.6 | |
| photo    | 92.5±0.6  | | | 86.9±1.9 | |