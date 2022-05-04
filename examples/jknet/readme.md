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
| cora | max | 0.847 | 0.01 | backend=paddle, lr=0.01, n_epoch=170, hidden_dim=32, drop_rate=0.5, itera_K=6, self_loops=1, weight_decay=0.001 |
| pubmed | concat | 0.7782 | 0.003 | backend=paddle, lr=0.01, n_epoch=300, hidden_dim=64, drop_rate=0.5, itera_K=4, self_loops=1, weight_decay=0.001 |
| citeseer | max | 0.7554 | 0.0013 | backend=paddle, lr=0.01, n_epoch=200, hidden_dim=64, drop_rate=0.5, itera_K=6, self_loops=1, weight_decay=0.01 |
