Simple Graph Convolution (SGC)
============

- Paper link: [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153)
- Author's code repo: [https://github.com/Tiiiger/SGC](https://github.com/Tiiiger/SGC). 


Results
-------

Run with following (available dataset: "cora", "citeseer", "pubmed")

```bash
python3 sgc_trainer.py
```

| dataset  | paper     | our(tf)     |
| -------- | --------- | ----------- |
| cora     | 81.0(0)   | 81.45(0.37) |
| citeseer | 71.9(0.1) | 69.03(0.27) |
| pubmed   | 78.9(0)   | 79.1(0)     |

