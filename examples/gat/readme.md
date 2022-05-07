Graph Attention Networks (GAT)
============

- Paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code repo (in Tensorflow):
  [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).
- Popular pytorch implementation:
  [https://github.com/Diego999/pyGAT](https://github.com/Diego999/pyGAT).


How to run
----------

Run with following:

```bash
python gat_trainer.py
```


Results
-------

| Dataset | Paper | Our(pd) | Our(tf) |
| ---- | ---- | ---- | ---- |
| cora |83.0(0.7)|83.54(0.75)|83.26(0.96)|
| pubmed |72.5(0.7)|72.74(0.76)|72.5(0.65)|
| citeseer |79.0(0.3)|OOM|OOM|
