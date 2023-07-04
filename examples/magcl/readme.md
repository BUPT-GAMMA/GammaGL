# GammaGL Implementation of MA-GCL
This GammaGL example implements the model proposed in the paper [MA-GCL: Model Augmentation Tricks for Graph Contrastive Learning.](https://arxiv.org/pdf/2212.07035.pdf).

Author's code: https://github.com/GXM1141/MA-GCL

## Example Implementor

This example was implemented by LiZhiwei

## Datasets

##### Unsupervised Node Classification Datasets:

'Cora', 'Citeseer' and 'Pubmed'

| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |


## Arguments

```
--dataset               str     The graph dataset name.                Default is 'cora'.
--n_epoch               int     Number of training periods.            Default is 500.
--lr                    float   Learning rate.                         Default is 0.0005.
--l2                    float   Weight decay.                          Default is 0.00001.
--temp                  float   Temperature.                           Default is 0.4.
--hid_dim               int     Hidden dimension.                      Default is 128.
--out_dim               int     Output dimension.                      Default is 128.
--num_layers            int     Number of GNN layers.                  Default is 2.
--drop_edge_rate_1      float   Drop edge ratio 1.                     Default is 0.2. 
--drop_edge_rate_2      float   Drop edge ratio 2.                     Default is 0.4. 
--drop_feature_rate_1   float   Drop feature ratio 1.                  Default is 0.3. 
--drop_feature_rate_2   float   Drop feature ratio 2.                  Default is 0.4. 
--dataset_path          str     path to save dataset.                  Default is r'../'
--best_model_path       str     path to save best model                Default is r'./'
```

## How to run examples

In the paper(as well as authors' repo), the training set and testing set are split randomly with 1:9 ratio. In order to fairly compare it with other methods with the public split (20 training nodes each class), in this repo we also provide its results using the public split (with fine-tuned hyper-parameters). To run the examples, follow the following instructions.

```bash
# use paddle backend

# Cora with random split
TL_BACKEND=paddle python magcl_trainer.py --dataset cora --epochs 500 --drop_edge_rate_1 0.4 --drop_edge_rate_2 0.2 --drop_feature_rate_1 0.5 --drop_feature_rate_2 0.5 --lr 0.0005
# Citeseer with random split
TL_BACKEND=paddle python magcl_trainer.py --dataset citeseer --epochs 600 --drop_edge_rate_1 0.4 --drop_edge_rate_2 0.4 --drop_feature_rate_1 0.3 --drop_feature_rate_2 0.3 --lr 0.0001
```
```bash
# use tensorflow backend

# Cora with random split
TL_BACKEND=tensorflow python magcl_trainer.py --dataset cora --epochs 500 --drop_edge_rate_1 0.4 --drop_edge_rate_2 0.2 --drop_feature_rate_1 0.5 --drop_feature_rate_2 0.5 --lr 0.0005
# Citeseer with random split
TL_BACKEND=tensorflow python magcl_trainer.py --dataset citeseer --epochs 600 --drop_edge_rate_1 0.4 --drop_edge_rate_2 0.4 --drop_feature_rate_1 0.3 --drop_feature_rate_2 0.3 --lr 0.0001

```
```bash 
# use pytorch backend

# Cora with random split
TL_BACKEND=torch python magcl_trainer.py --dataset cora --epochs 500 --drop_edge_rate_1 0.4 --drop_edge_rate_2 0.2 --drop_feature_rate_1 0.5 --drop_feature_rate_2 0.5 --lr 0.0005
# Citeseer with random split
TL_BACKEND=torch python magcl_trainer.py --dataset --dataset citeseer --epochs 600 --drop_edge_rate_1 0.4 --drop_edge_rate_2 0.4 --drop_feature_rate_1 0.3 --drop_feature_rate_2 0.3 --lr 0.0001


```

## 	Performance




|      Dataset      | Cora | Citeseer | Pubmed |
| :---------------: |:----:|:--------:|:------:|
|   Author's Code   | 83.3 |   73.6   |  83.5  |
|     GammaGL(tf)   | 82.1 |   72.0   |   -    |
|     GammaGL(th)   | 80.5 |   71.4   |   -    |
|     GammaGL(pd)   | 82.4 |   70.4   |   -    |



