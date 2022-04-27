Representation Learning on Graphs with Jumping Knowledge Networks (JK-Net)
============

- Paper link: [https://arxiv.org/abs/1806.03536](https://arxiv.org/abs/1806.03536)
- Author's code repo: [https://github.com/ShinKyuY/Representation_Learning_on_Graphs_with_Jumping_Knowledge_Networks](https://github.com/ShinKyuY/Representation_Learning_on_Graphs_with_Jumping_Knowledge_Networks). Note that the original code is 
implemented with Tensorflow for the paper. 

Structure
-------
![img.png](img.png)

Results
-------

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python jknet_trainer.py --dataset cora --params
python jknet_trainer.py --dataset citeseer 
python jknet_trainer.py --dataset cora
```


| Dataset | mode | Accuracy | std | Params |
| ---- | ---- | ---- | ---- | ---- |
| cora | max | 0.7758 | 0.01 | backend=paddle, lr=0.005, n_epoch=400, hidden_dim=64, drop_rate=0.5, itera_K=6, self_loops=1, weight_decay=0.001 |
| pubmed | concat | 0.7782 | 0.003 | backend=paddle, lr=0.01, n_epoch=300, hidden_dim=64, drop_rate=0.5, itera_K=4, self_loops=1, weight_decay=0.001 |
| citeseer | concat | 0.6909 | 0.01 | backend=paddle, lr=0.01, n_epoch=150, hidden_dim=64, drop_rate=0.6, itera_K=8, self_loops=1, weight_decay=0.01 |
