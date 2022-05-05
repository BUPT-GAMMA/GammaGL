Graph Convolutional Networks (GCN)
============

- Paper link: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
- Author's code repo: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn). Note that the original code is 
implemented with Tensorflow for the paper. 


Results
-------

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python gcn_train.py --dataset cora
python gcn_train.py --dataset citeseer 
python gcn_train.py --dataset pubmend
```


| Dataset  | Paper | Our(pd)    | Our(tf) |
|----------|-------|-------------|---------|
| cora     | 81.5  | 81.83(0.22) |80.54(1.12)|
| pubmed   | 70.3  | 68.9(0.02) |68.34(0.68)|
| citeseer | 79.0  | 78.29(0.01) |78.28(1.08)|

