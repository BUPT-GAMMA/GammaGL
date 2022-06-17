# GammaGL Implementation of MERIT
This GammaGL example implements the model proposed in the paper [Multi-Scale Contrastive Siamese Networks for Self-Supervised Graph Representation Learning](https://arxiv.org/abs/2105.05682).

Author's code: https://github.com/GRAND-Lab/MERIT

## Example Implementor

This example was implemented by Ziyu Zheng

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
--input_dim 			int		Input dimension.                       Default is 1433.
--out_dim				int		Output dimension.					   Default is 512.
--proj_size 			int		Encoder output dimension			   Default is 512.
--proj_hid 				int     Encoder hidden dimension			   Default is 4096.
--pred_size 			int		MLP output dimension			       Default is 512.
--pred_hid 				int		MLP hidden dimension			   	   Default is 4096.
--drop_edge_rate_1      float   Drop edge ratio 1.                     Default is 0.2. 
--drop_edge_rate_2      float   Drop edge ratio 2.                     Default is 0.2. 
--drop_feature_rate_1   float   Drop feature ratio 1.                  Default is 0.5. 
--drop_feature_rate_2   float   Drop feature ratio 2.                  Default is 0.5. 
--dataset_path          str     path to save dataset.                  Default is r'../'
```

## How to run examples

In the paper(as well as authors' repo), the training set are full graph training

```python
# use paddle backend
# Cora by GammaGL
TL_BACKEND=paddle python merit_trainer.py --dataset cora --epochs 500 --drop_edge_rate_1 0.2 --drop_edge_rate_2 0.2 --drop_feature_rate_1 0.5 --drop_feature_rate_2 0.5 --lr 3e-4 --beta 0.5
#Citeseer by GammaGL
TL_BACKEND=paddle python merit_trainer.py --dataset citeseer --epochs 500 --drop_edge_rate_1 0.4 --drop_edge_rate_2 0.4 --drop_feature_rate_1 0.5 --drop_feature_rate_2 0.5 --lr 3e-4 --beta 0.6

# use tensorflow backend
# Cora by GammaGL
TL_BACKEND=tensorflow python merit_trainer.py --dataset cora --epochs 500 --drop_edge_rate_1 0.2 --drop_edge_rate_2 0.2 --drop_feature_rate_1 0.5 --drop_feature_rate_2 0.5 --lr 3e-4 --beta 0.5
#Citeseer by GammaGL
TL_BACKEND=tensorflow python merit_trainer.py --dataset citeseer --epochs 500 --drop_edge_rate_1 0.4 --drop_edge_rate_2 0.4 --drop_feature_rate_1 0.5 --drop_feature_rate_2 0.5 --lr 3e-4 --beta 0.6


```

## 	Performance


|     Dataset     | Cora | Citeseer | Pubmed |
| :-------------: | :--: | :------: | :----: |
|  Author's Code  | 83.1 |   74.0   |  80.2  |
|   GammaGL(tf)   | 84.3 |   72.2   |  --.-  |
| GammaGL(paddle) | 83.1 |   --.-   |  --.-  |

 