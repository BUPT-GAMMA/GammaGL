Graph Convolutional Networks (GCN)
============

- Paper link: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
- Author's code repo: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn). Note that the original code is 
implemented with Tensorflow for the paper. 


Results
-------

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python gcn_train.py --dataset cora --params
python gcn_train.py --dataset citeseer 
python gcn_train.py --dataset cora
```


| Dataset | Paper | Accuracy | Time | Params |
| ---- | ---- | ---- | ---- | ---- |
| cora | 81.5 |81.83(0.22) | | backend=paddle, lr=0.01, l2_coef=5e-3, drop_rate=0.9|
| pubmed | 70.3 |
| citeseer | 79.0 |

