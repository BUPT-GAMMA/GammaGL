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
TL_BACKEND=tensorflow python train_full.py --dataset cora --lr 0.01 --hidden_dim 128 --drop_rate 0.7 --n_epoch 500
TL_BACKEND=tensorflow python reddit_sage.py --lr 0.0005 --hidden_dim 256 --drop_rate 0.8
```
```bash
TL_BACKEND=paddle python train_full.py --dataset cora --n_epoch 500 --lr 0.005 --hidden_dim 512 --drop_rate 0.7 --n_epoch 500
CUDA_VISIBLE_DEVICES=5 TL_BACKEND=paddle python reddit_sage.py --lr 0.001 --hidden_dim 128 --drop_rate 0.8
```
```bash
# use pytorch
TL_BACKEND=torch python train_full.py --dataset cora --n_epoch 500 --lr 0.005 --hidden_dim 512 --drop_rate 0.8
TL_BACKEND=torch python reddit_sage.py --lr 0.005 --hidden_dim 128 --drop_rate 0.8
```


|      Dataset      |      Cora         | Reddit |
| :---------------: | :---------------: | :----: |
|        DGL        |       83.3        | 94.95 |
|       Paper       |       83.3        | 95.0  |
|     GammaGL(tf)   |    82.44 ± 0.88   | 95.0  |
|     GammaGL(th)   |    81.13 ± 1.08   | 94.9  |
|     GammaGL(pd)   |    82.04 ± 0.33   | 91.2  |
|     GammaGL(ms)   |        --.-       | --.-  |

* We fail to reproduce the reported accuracy on 'Cora', even with the DGL's code.
* The model performance is the average of 5 tests
