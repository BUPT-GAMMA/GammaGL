Inductive Representation Learning on Large Graphs (GraphSAGE)
============

- Paper link: [http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)
- Author's code repo: [https://github.com/williamleif/graphsage-simple](https://github.com/williamleif/graphsage-simple). Note that the original code is 
simple reference implementation of GraphSAGE.


Results
-------

### Full graph training

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python train_full.py --dataset cora     # full graph
```

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python train_full.py --dataset cora 
python train_full.py --dataset citeseer 
python train_full.py --dataset pubmed
```


* cora: ~0.81530 
* citeseer: ~0.6930
* pubmed: ~0.7830
