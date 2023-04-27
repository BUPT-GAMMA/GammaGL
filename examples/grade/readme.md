# GammaGL Implementation of GRADE
This GammaGL example implements the model proposed in the paper ["**Uncovering the Structural Fairness in Graph Contrastive Learning**"](https://arxiv.org/abs/2210.03011).

Author's code: https://github.com/BUPT-GAMMA/Uncovering-the-Structural-Fairness-in-Graph-Contrastive-Learning

## Example Implementor

This example was implemented by Yifei Shao

## Datasets

'Cora', 'Citeseer' 'Photo' and 'Computers'

| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Photo    | 7,650   | 238,162 | 8         |
| Computers| 13,752  | 491,722 | 10        |


## Arguments

```
--gpu_id                int     Set device for training.               Default is 0.
--dataset               str     The graph dataset name.                Default is 'cora'.
--warmup                int     Warmup of training.                    Default is 200.
--epoch                 int     Number of training periods.            Default is 400.
--lr                    float   Learning rate.                         Default is 0.001.
--wd                    float   Weight decay.                          Default is 1e-5.
--threshold             int     Definition of low-degree nodes.        Default is 9.
--act_fn                str     Activation function                    Default is 'relu'.
--temp                  float   Temperature.                           Default is 0.5.
--hid_dim               int     Hidden dimension.                      Default is 256.
--out_dim               int     Output dimension.                      Default is 256.
--num_layers            int     Number of GNN layers.                  Default is 2.
--der1                  float   Drop edge ratio 1.                     Default is 0.2. 
--der2                  float   Drop edge ratio 2.                     Default is 0.2. 
--dfr1                  float   Drop feature ratio 1.                  Default is 0.2. 
--dfr2                  float   Drop feature ratio 2.                  Default is 0.2. 
--dataset_path          str     Path to save dataset.                  Default is r'../'
--mode                  str     Split dataset.                         Default is 'full'.
```

## How to run examples

```bash
# use paddle backend

# Cora 
TL_BACKEND=paddle python grade_trainer.py --dataset cora --mode full --hid_dim 256 --out_dim 256 --act_fn relu --temp 0.8 --gpu_id (up to you)
# Citeseer
TL_BACKEND=paddle python grade_trainer.py --dataset citeseer --mode full --hid_dim 256 --out_dim 256 --act_fn relu --temp 1.7 --gpu_id (up to you)
# Photo
TL_BACKEND=paddle python grade_trainer.py --dataset photo --mode full --hid_dim 512 --out_dim 512 --act_fn relu --temp 0.8 --gpu_id (up to you)
# Computers
TL_BACKEND=paddle python grade_trainer.py --dataset computers --mode full --hid_dim 800 --out_dim 800 --act_fn prelu --temp 1.1 --gpu_id (up to you)
```
```bash
# use tensorflow backend

# Cora 
TL_BACKEND=tensorflow python grade_trainer.py --dataset cora --mode full --hid_dim 256 --out_dim 256 --act_fn relu --temp 0.8 --gpu_id (up to you)
# Citeseer
TL_BACKEND=tensorflow python grade_trainer.py --dataset citeseer --mode full --hid_dim 256 --out_dim 256 --act_fn relu --temp 1.7 --gpu_id (up to you)
# Photo
TL_BACKEND=tensorflow python grade_trainer.py --dataset photo --mode full --hid_dim 512 --out_dim 512 --act_fn relu --temp 0.8 --gpu_id (up to you)
# Computers
TL_BACKEND=tensorflow python grade_trainer.py --dataset computers --mode full --hid_dim 800 --out_dim 800 --act_fn prelu --temp 1.1 --gpu_id (up to you)

```
```bash 
# use pytorch backend

# Cora 
TL_BACKEND=torch python grade_trainer.py --dataset cora --mode full --hid_dim 256 --out_dim 256 --act_fn relu --temp 0.8 --gpu_id (up to you)
# Citeseer
TL_BACKEND=torch python grade_trainer.py --dataset citeseer --mode full --hid_dim 256 --out_dim 256 --act_fn relu --temp 1.7 --gpu_id (up to you)
# Photo
TL_BACKEND=torch python grade_trainer.py --dataset photo --mode full --hid_dim 512 --out_dim 512 --act_fn relu --temp 0.8 --gpu_id (up to you)
# Computers
TL_BACKEND=torch python grade_trainer.py --dataset computers --mode full --hid_dim 800 --out_dim 800 --act_fn prelu --temp 1.1 --gpu_id (up to you)


```
## Note!
* In the original paper, the first 1000 nodes with degrees greater than 0 and less than 50 are selected as the testing set. However, since GammaGL and DGL use different node order when loading the same dataset,
the nodes in the testing set are not the same. Since the partitioning of datasets will have a great impact on the evaluation of model performance, we fail to reproduce the same performance mentioned in the paper. For the same reason, the performance can not be reproduced by using pytorch_geometric.
* Because tensorlayerx lacks some operations(e.g. torch.multinomial, torch.repeat_interleave), We use numpy instead. Therefore, it takes longer time to train our model than the author's code.
* We recommend using the author's code for model comparison. https://github.com/BUPT-GAMMA/Uncovering-the-Structural-Fairness-in-Graph-Contrastive-Learning
## 	Performance

| Evaluation(mode=full) |   F1Mi   |   F1Ma   |   Mean   |   Bias   |
| :-------------------: | :------: | :------: | :------: | :------: |
|   Cora(paper)         |  0.8340  |  0.7854  |  0.9287  |  0.0048  |
|   Cora(ours)          |  0.8660  |  0.8604  |  0.9027  |  0.0378  |
|   Citeseer(paper)     |  0.6714  |  0.6104  |  0.8588  |  0.0152  |
|   Citeseer(ours)      |  0.7450  |  0.7045  |  0.8724  |  0.0156  |
|   Photo(paper)        |  0.9472  |  0.7886  |  0.9852  |  0.0020  |
|   Photo(ours)         |  0.9030  |  0.8980  |  0.9199  |  0.0050  |
|   Computers(paper)    |  0.8942  |  0.7471  |  0.9742  |  0.0035  |
|   Computers(ours)     |  0.8760  |  0.8740  |  0.8880  |  0.0061  |

