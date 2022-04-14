Deep Graph Infomax (DGI)
========================

- Paper link: [https://arxiv.org/abs/1809.10341](https://arxiv.org/abs/1809.10341)
- Author's code repo (in Pytorch):
  [https://github.com/PetarV-/DGI](https://github.com/PetarV-/DGI)


How to run
----------

Run with following:

```bash
python3 train.py --dataset=cora --hidden_dim=512 --lr=0.001 --n_epoch=500
```

```bash
python3 train.py --dataset=citeseer 
```

```bash
python3 train.py --dataset=pubmed --hidden_dim=256 --lr=0.002 --n_epoch=500
```

Results
-------

=======
|      Dataset      | Cora | Citeseer | Pubmed |
| :---------------: | :--: | :------: | :----: |
|     GammaGL(tf)   | 82.1 |   69.2   |  78.85 |
|     GammaGL(th)   | --.- |   --.-   |  --.- |
|     GammaGL(pd)   | --.- |   --.-   |  --.- |
|     GammaGL(ms)   | --.- |   --.-   |  --.- |
|   Author's Code   | 82.3 |   71.8   |  76.8  |
|        DGL        | 81.6 |   69.4   |  76.1  |




