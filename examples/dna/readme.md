# JUST JUMP: DYNAMIC NEIGHBORHOOD AGGREGATION IN GRAPH NEURAL NETWORKS (DNA)

- Paper link: [https://arxiv.org/abs/1904.04849](https://arxiv.org/abs/1904.04849)
- The implementation of PyG: [https://github.com/pyg-team/pytorch_geometric/blob/master/examples/dna.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/dna.py). 

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

```

| Dataset  | group | Paper      | Our(pd) | Our(tf) | Our(th)    | Our(ms) |
| -------- | ----- | ---------- | ------- | ------- | ---------- | ------- |
| cora     | 1     | 83.88±0.50 |         |         | 81.43±0.17 |         |
| cora     | 8     | 85.86±0.45 |         |         |            |         |
| cora     | 16    | 86.15±0.57 |         |         |            |         |
| citeseer | 1     | 73.37±0.83 |         |         | 70.53±0.18 |         |
| citeseer | 8     | 74.19±0.66 |         |         |            |         |
| citeseer | 16    | 74.50±0.62 |         |         |            |         |
| pubmed   | 1     | 87.80±0.25 |         |         | 78.63±0.12 |         |
| pubmed   | 8     | 88.04±0.17 |         |         |            |         |
| pubmed   | 16    | 88.04±0.22 |         |         |            |         |
