Graph Convolutional Networks (GCN)
============

- Paper link: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
- Author's code repo: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn). Note that the original code is 
implemented with Tensorflow for the paper. 


Results
-------

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
pytho3 gcn_train.py
```


| Dataset | Paper | Accuracy | Time | Params |
| ---- | ---- | ---- | ---- | ---- |
| cora | 81.5 |
| pubmed | 70.3 |
| citeseer | 79.0 |