Principal Neighbourhood Aggregation for Graph Nets (PNA)
============

- Paper link: [https://grlplus.github.io/papers/20.pdf](https://grlplus.github.io/papers/20.pdf)
- Author's code repo (in PyTorch):
  [https://github.com/lukecavabarrett/pna](https://github.com/lukecavabarrett/pna).

Dataset
-------

The ZINC dataset from the [ZINC database](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559) and the
[Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules](https://arxiv.org/abs/1610.02415) 
paper, containing about 250,000 molecular graphs with up to 38 heavy atoms. Our experiments only load a subset of the 
dataset (12,000 molecular graphs), following the [Benchmarking Graph Neural Networks](https://arxiv.org/abs/2003.00982)
paper.

Results from the Paper
-------

| Task             | Dataset | Model | Metric Name | Metric Value |
|------------------|---------|-------|-------------|--------------|
| Graph Regression | ZINC    | PNA   | MAE         | 0.188±0.004  |


Our Results
-----------

```bash
TL_BACKEND="paddle" python pna_trainer.py --batch_size 128 --lr 0.001 --n_epoch 400
TL_BACKEND="torch" python pna_trainer.py --batch_size 128 --lr 0.001 --n_epoch 400
TL_BACKEND="tensorflow" python pna_trainer.py --batch_size 128 --lr 0.001 --n_epoch 400
```

| Dataset | Our(pd)  | Our(torch) | Our(tf)       |
|---------|----------|------------|---------------|
| ZINC    | OOM      | 0.186      | 0.195(±0.006) |
