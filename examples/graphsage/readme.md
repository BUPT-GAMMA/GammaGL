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
# use Tensorflow
TL_BACKEND="tensorflow" python train_full.py --dataset cora --n_epoch 100 --lr 0.001 --hidden_dim 512
TL_BACKEND="tensorflow" python train_full.py --dataset citeseer --n_epoch 100 --lr 0.001 --hidden_dim 512 
TL_BACKEND="tensorflow" python train_full.py --dataset pubmed --n_epoch 200 --lr 0.01 --hidden_dim 16
TL_BACKEND="tensorflow" python reddit_sage.py 

# use Paddle
TL_BACKEND="paddle" python train_full.py --dataset cora --n_epoch 100 --lr 0.001 --hidden_dim 512
TL_BACKEND="paddle" python train_full.py --dataset citeseer --n_epoch 100 --lr 0.001 --hidden_dim 512 
TL_BACKEND="paddle" python train_full.py --dataset pubmed --n_epoch 200 --lr 0.01 --hidden_dim 16
TL_BACKEND="paddle" python reddit_sage.py 
```


|      Dataset      | Cora | Citeseer | Pubmed | Reddit |
| :---------------: | :--: | :------: | :----: | :----: |
|        DGL        | 83.3 |   71.1   |  78.3  |  94.95 |
|       Paper       | 83.3 |   --.-   |  --.-  |  95.0  |
|     GammaGL(tf)   | 81.7 |   70.4   |  --.-  |  94.8  |
|     GammaGL(th)   | --.- |   --.-   |  --.-  |  --.-  |
|     GammaGL(pd)   | 81.7 |   69.7   |  --.-  |  94.8  |
|     GammaGL(ms)   | --.- |   --.-   |  --.-  |  --.-  |
