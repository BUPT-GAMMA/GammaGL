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
# use fensorflow background
TL_BACKEND="tensorflow" python train_full.py --dataset cora --lr 0.0005 --hidden_dim 256 --drop_rate 0.8
TL_BACKEND="tensorflow" python reddit_sage.py 
```
```bash
# use paddle
TL_BACKEND="paddle" python train_full.py --dataset cora --n_epoch 100 --lr 0.001 --hidden_dim 512
TL_BACKEND="paddle" python reddit_sage.py 
```
```bash
# use pytorch
TL_BACKEND="torch" python train_full.py --dataset cora --n_epoch 100 --lr 0.001 --hidden_dim 512
TL_BACKEND="torch" python reddit_sage.py 
```


|      Dataset      | Cora | Reddit |
| :---------------: | :---------------: | :----: |
|        DGL        | 83.3 |94.95 |
|       Paper       | 83.3 |95.0  |
|     GammaGL(tf)   | 82.2 | 94.5  |
|     GammaGL(th)   | --.- | --.-  |
|     GammaGL(pd)   | 81.7 | 94.8  |
|     GammaGL(ms)   | --.- | --.-  |
