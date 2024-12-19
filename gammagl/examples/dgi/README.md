Deep Graph Infomax (DGI)
========================

- Paper link: [https://arxiv.org/abs/1809.10341](https://arxiv.org/abs/1809.10341)
- Author's code repo (in Pytorch):
  [https://github.com/PetarV-/DGI](https://github.com/PetarV-/DGI)


How to run
----------

Run with following:

```bash
# use tensorflow backend
TL_BACKEND=paddle python dgi_trainer.py --dataset cora --lr 0.002 --patience 50
TL_BACKEND=paddle python dgi_trainer.py --dataset citeseer --lr 0.0005 --patience 20 --n_epoch 300
TL_BACKEND=paddle python dgi_trainer.py --dataset pubmed --lr 0.001 --hidden_dim 256 --patience 20 
```

```bash
# use paddle backend
TL_BACKEND=tensorflow python dgi_trainer.py --dataset cora --lr 0.003 --patience 50
TL_BACKEND=tensorflow python dgi_trainer.py --dataset citeseer --lr 0.001 --patience 20 --n_epoch 100
TL_BACKEND=tensorflow python dgi_trainer.py --dataset pubmed --hidden_dim 256 --lr 0.001
```
```bash
# use pytorch backend
TL_BACKEND=torch python dgi_trainer.py --dataset cora 
TL_BACKEND=torch python dgi_trainer.py --dataset citeseer
TL_BACKEND=torch python dgi_trainer.py --dataset pubmed --lr 0.001 --patience 20
```

Results
-------


|      Dataset      |     Cora     |    Citeseer    |   Pubmed   |
| :---------------: | :----------: | :------------: | :--------: |
|   Author's Code   | 82.3         |   71.8         |  76.8         |
|        DGL        | 81.6         |   69.4         |  76.1         |
|     GammaGL(tf)   | 81.51 ± 0.55 |  69.01 ± 0.91 | 78.37 ± 0.37 |
|     GammaGL(th)   | --.-         |   --.-         | 79.58 ± 0.52 |
|     GammaGL(pd)   | 81.19 ± 0.64 | 69.06 ± 0.50 | 78.58 ± 0.65 |
|     GammaGL(ms)   | --.-         |   --.-         |  --.-  |

* The model performance is the average of 5 tests