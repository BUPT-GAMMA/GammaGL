ClusterGCN
============

- Paper link: [https://arxiv.org/abs/1905.07953](https://arxiv.org/abs/1905.07953)
- Author's code repo (in Tensorflow):
  [https://github.com/google-research/google-research/tree/master/cluster_gcn](https://github.com/google-research/google-research/tree/master/cluster_gcn).

Dataset Statics
-------

| Dataset | # Nodes | # Edges    | # Classes |
|---------|---------|------------|-----------|
| ppi     | 56,944  | 818,716    | 121       |
| reddit  | 232,965 | 11,606,919 | 41        |


Results
-------

```bash
L_BACKEND="tensorflow" python clustergcn_ppi_trainer.py
L_BACKEND="torch" python clustergcn_ppi_trainer.py
L_BACKEND="tensorflow" python clustergcn_reddit_trainer.py
L_BACKEND="torch" python cclustergcn_reddit_trainer.py
```

| Dataset | Paper | Our(torch) | Our(tf) |
|---------|-------|------------|---------|
| ppi     | 99.3  | 99.2       | 99.3    |
| reddit  | 96.6  | 99.5       | 99.6    |

