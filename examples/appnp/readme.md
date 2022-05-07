Predict then Propagate: Graph Neural Networks meet Personalized PageRank (APPNP)
============

- Paper link: [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997)

- Author's code repo:https://github.com/gasteigerjo/ppnp). 

> This example does not contain the implementation of PPNP.

Results
-------

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 appnp_trainer.py
```

| dataset  | paper        | our(pd)    | our(tf)    |
| -------- | ------------ | ---------- | ---------- |
| cora     |              | 82.9(0.56) | 76.3(0.45) |
| citeseer | 75.83 ± 0.27 | 79.8(0.67) | 75.5(0)    |
| pubmed   | 79.73 ± 0.31 | 70.2(0.8)  | 65.7(0.15) |
| cora-ml  | 85.29 ± 0.25 |            |            |

