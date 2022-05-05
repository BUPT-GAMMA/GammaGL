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


| Dataset | Accuracy | std | Paper |
| ---- | ---- | ---- | ---- |
| cora | 0.847 | 0.01 | 0.896(0.005) |
| pubmed | 0.7782 | 0.003 | |
| citeseer | 0.7554 | 0.0013 | 0.783(0.008)|
