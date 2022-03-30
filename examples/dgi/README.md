Deep Graph Infomax (DGI)
========================

- Paper link: [https://arxiv.org/abs/1809.10341](https://arxiv.org/abs/1809.10341)
- Author's code repo (in Pytorch):
  [https://github.com/PetarV-/DGI](https://github.com/PetarV-/DGI)


How to run
----------

Run with following:

```bash
python3 train.py --dataset=cora --hidden_dim=512 --lr=0.001 --n_epoch=200
```

```bash
python3 train.py --dataset=citeseer 
```

```bash
python3 train.py --dataset=pubmed --hidden_dim=256 --lr=0.002 --n_epoch=500
```

Results
-------
* cora: ~82.1 (81.8-82.4) (paper: 82.3)
* citeseer: ~69.2(68.7-69.5) (paper: 71.8)
* pubmed: ~78.85(78.2-79.5) (paper: 76.8)
