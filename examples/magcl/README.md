# GammaGL Implementation of MA-GCL
This GammaGL example implements the model proposed in the paper [Deep Graph Contrastive Representation Learning](https://arxiv.org/abs/2006.04131).

Author's code: https://github.com/GXM1141/MA-GCL

## Example Implementor

This example was implemented by Zhiwei Le

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
TL_BACKEND=paddle python magcl_trainer.py --dataset cora --n_epoch 100
# Citeseer with random split
TL_BACKEND=paddle python magcl_trainer.py --dataset citeseer --n_epoch 20
```
```bash
# use tensorflow backend

# Cora with random split
TL_BACKEND=tensorflow python magcl_trainer.py --dataset cora --n_epoch 150
# Citeseer with random split
TL_BACKEND=tensorflow python magcl_trainer.py --dataset citeseer --lr 2e-3 --n_epoch 75 --hid_dim 256

```
```bash 
# use pytorch backend

# Cora with random split
TL_BACKEND=torch python magcl_trainer.py --dataset cora --n_epoch 500
# Citeseer with random split
TL_BACKEND=torch python magcl_trainer.py --dataset citeseer --n_epoch 200 --lr 1e-3 --l2 1e-5 --hid_dim 256 --drop_edge_rate_1 0.2 --drop_edge_rate_2 0.0 --drop_feature_rate_1 0.3 --drop_feature_rate_2 0.2 --temp 0.9


```

## 	Performance




|      Dataset      | Cora | Citeseer | Pubmed |
| :---------------: |:----:|:--------:|:------:|
|   Author's Code   | 83.3 |   73.6   |  83.5  |
|     GammaGL(tf)   | 82.1 |   72.0   |   -    |
|     GammaGL(th)   | 80.5 |   71.4   |   -    |
|     GammaGL(pd)   |  -   |    -     |   -    |

* I can't run the model with paddle on my laptop, I will update acc of it soon.
* I can't run the model on Pubmed on my laptop, I will continue work on it.
* The acc on ggl can't reach the acc from the Author's code, I will try to improve it soon.
