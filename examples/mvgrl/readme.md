# GammaGL Implementation of MVGRL
This GammaGL example implements the model proposed in the paper [Contrastive Multi-View Representation Learning on Graphs](https://arxiv.org/abs/2006.05582).

Author's code: https://github.com/kavehhassani/mvgrl


## Datasets

##### Unsupervised Graph Classification Datasets:

 'MUTAG', 'PTC_MR', 'REDDIT-BINARY', 'IMDB-BINARY', 'IMDB-MULTI'.

| Dataset         | MUTAG | PTC_MR | RDT-B  | IMDB-B | IMDB-M |
| --------------- | ----- | ------ | ------ | ------ | ------ |
| # Graphs        | 188   | 344    | 2000   | 1000   | 1500   |
| # Classes       | 2     | 2      | 2      | 2      | 3      |
| Avg. Graph Size | 17.93 | 14.29  | 429.63 | 19.77  | 13.00  |
* RDT-B, IMDB-B, IMDB-M are short for REDDIT-BINARY, IMDB-BINARY and IMDB-MULTI respectively.

##### Unsupervised Node Classification Datasets:

'Cora', 'Citeseer' and 'Pubmed'

| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |


## Arguments

##### 	Graph Classification:

```
--dataset         str     The graph dataset name.                Default is 'MUTAG'.
--n_epochs          int     Number of training periods.            Default is 200.
--patience         int     Early stopping steps.                  Default is 20.
--lr               float   Learning rate.                         Default is 0.001.
--l2_coef               float   Weight decay.                          Default is 0.0.
--batch_size       int     Size of a training batch.              Default is 64.
--num_layers         int     Number of GNN layers.                  Default is 4.
--hidden_dim          int     Embedding dimension.                   Default is 32.
```

##### 	Node Classification:

```
--dataset         str     The graph dataset name.                Default is 'cora'.
--n_epochs          int     Number of training periods.            Default is 500.
--patience         int     Early stopping steps.                  Default is 20.
--lr             float   Learning rate of main model.           Default is 0.001.
--clf_lr             float   Learning rate of linear classifer.     Default is 0.01.
--l2_coef              float   Weight decay of main model.            Default is 0.0.
--clf_l2_coef              float   Weight decay of linear classifier.     Default is 0.0.
--epsilon          float   Edge mask threshold.                   Default is 0.01.
--hidden_dim          int     Embedding dimension.                   Default is 512.
--sample_size      int     Subgraph size.                         Default is 2000.
```

## How to run examples

###### Graph Classification

```python
# Enter the 'graph' directory
cd graph

# MUTAG:
CUDA_VISIBLE_DEVICES="0" TL_BACKEND="paddle" python mvgrl_trainer.py --dataset MUTAG --n_epoch 20
CUDA_VISIBLE_DEVICES="0" TL_BACKEND="tensorflow" python mvgrl_trainer.py --dataset MUTAG --n_epoch 20
CUDA_VISIBLE_DEVICES="0" TL_BACKEND="torch" python mvgrl_trainer.py --dataset MUTAG --n_epoch 20

# PTC_MR:
CUDA_VISIBLE_DEVICES="0" TL_BACKEND="paddle" python mvgrl_trainer.py --dataset PTC_MR --epochs 32 --hid_dim 128
CUDA_VISIBLE_DEVICES="0" TL_BACKEND="tensorflow" python mvgrl_trainer.py --dataset PTC_MR --epochs 32 --hid_dim 128
CUDA_VISIBLE_DEVICES="0" TL_BACKEND="torch" python mvgrl_trainer.py --dataset PTC_MR --epochs 32 --hid_dim 128

# IMDB-BINARY

CUDA_VISIBLE_DEVICES="0" TL_BACKEND="paddle"  python mvgrl_trainer.py --dataset IMDB-BINARY --epochs 20 --hid_dim 512 --n_layers 2
CUDA_VISIBLE_DEVICES="0" TL_BACKEND="tensorflow" python mvgrl_trainer.py --dataset IMDB-BINARY --epochs 20 --hid_dim 512 --n_layers 2
CUDA_VISIBLE_DEVICES="0" TL_BACKEND="torch" python mvgrl_trainer.py --dataset IMDB-BINARY --epochs 20 --hid_dim 512 --n_layers 2
# IMDB-MULTI
CUDA_VISIBLE_DEVICES="0" TL_BACKEND="paddle" python mvgrl_trainer.py --dataset IMDB-MULTI --epochs 20 --hid_dim 512 --n_layers 2
CUDA_VISIBLE_DEVICES="0" TL_BACKEND="tensorflow" python mvgrl_trainer.py --dataset IMDB-MULTI --epochs 20 --hid_dim 512 --n_layers 2
CUDA_VISIBLE_DEVICES="0" TL_BACKEND="torch" python mvgrl_trainer.py --dataset IMDB-MULTI --epochs 20 --hid_dim 512 --n_layers 2

```
###### Node Classification

For semi-supervised node classification on 'Cora', 'Citeseer'

full-graph training, see 'mvgrl_trainer.py', where we contrast the local and global representations of the whole graph.

```python
# Enter the 'node' directory
cd node

# Cora with full graph
CUDA_VISIBLE_DEVICES="0" TL_BACKEND="paddle" python mvgrl_trainer.py --dataset cora 
CUDA_VISIBLE_DEVICES="0" TL_BACKEND="tensorflow" python mvgrl_trainer.py --dataset cora 
CUDA_VISIBLE_DEVICES="0" TL_BACKEND="torch" python mvgrl_trainer.py --dataset cora 

# Citeseer 
CUDA_VISIBLE_DEVICES="0" TL_BACKEND="paddle" python mvgrl_trainer.py --dataset citeseer
CUDA_VISIBLE_DEVICES="0" TL_BACKEND="tensorflow" python mvgrl_trainer.py --dataset citeseer
CUDA_VISIBLE_DEVICES="0" TL_BACKEND="torch" python mvgrl_trainer.py --dataset citeseer
```

## 	Performance

We use the same  hyper-parameter settings as stated in the original paper.

##### Graph classification:

|      Dataset      | MUTAG | PTC-MR | REDDIT-B | IMDB-B | IMDB-M |
| :---------------: | :---: | :----: | :------: | :----: | :----: |
| Accuracy Reported | 89.7  |  62.5  |   84.5   |  74.2  |  51.2  |
|        DGL        | 89.4  |  62.2  |   85.0   |  73.8  |  51.1  |
|    GammaGL(tf)    | 89.4  |  60.0  |   OOM    |  74.0  |  50.5  |
|    GammaGL(th)    | --.-  |  --.-  |   --.-   |  --.-  |  --.-  |
|    GammaGL(pd)    | --.-  |  --.-  |   --.-   |  --.-  |  --.-  |
|    GammaGL(ms)    | --.-  |  --.-  |   --.-   |  --.-  |  --.-  |

* The datasets that the authors used are slightly different from standard TUDataset (see gammagl.data.TUDataset) in the nodes' features(e.g. The node features of 'MUTAG' dataset are of dimensionality 11 rather than 7")

##### Node classification:



|      Dataset      | Cora | Citeseer | Pubmed |
| :---------------: | :--: | :------: | :----: |
| Accuracy Reported | 86.8 |   73.3   |  80.1  |
|    DGL-sample     | 83.2 |   72.6   |  79.8  |
|     DGL-full      | 83.5 |   73.7   |  OOM   |
|     GammaGL(tf)   | 81.64 ± 0.50 |   73.3   |  OOM   |
|     GammaGL(th)   | 82.8 |   73.9   |  OOM   |
|     GammaGL(pd)   | 82.5 |   --.-   |  OOM   |
|     GammaGL(ms)   | --.- |   --.-   |  OOM   |


* We fail to reproduce the reported accuracy on 'Cora', even with the authors' code.
* The accuracy reported by the original paper is based on fixed-sized subgraph-training.
* The model performance is the average of 3 tests