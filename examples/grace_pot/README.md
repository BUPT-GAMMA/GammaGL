# GammaGL Implementation of GRACE-POT
This GammaGL example implements the model proposed in the paper [xxxx](https://arxiv.org/).

Author's code:

## Example Implementor

This example was implemented by Siyuan Zhang

## Datasets

##### Unsupervised Node Classification Datasets:

'Cora', 'Citeseer' and 'Pubmed'

| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Pubmed   | 19,717  | 88,651  | 3         |
| Photo    | 7,650   | 238,162 | 8         |


## How to run examples
Fisrt, make the directories for datasets and bounds to save
``` bash
mkdir ~/datasets
mkdir ~/datasets/bounds
```
Then, go into the directory of a model. If you want to set the parameters, you should modify the ocnfiguration files in the directory ("config.yaml" for GRACE). The following is the command line to run each model (dataset used is Cora for example):
```bash
# original GRACE
python GRACE_POT_trainer.py --dataset Cora --gpu_id 0
# GRACE + POT
python GRACE_POT_trainer.py --dataset Cora --gpu_id 0 --use_pot --kappa 0.4
```
The result will be appended to the file "res/{dataset_name}_base_temp.csv" and "res/{dataset_name}_pot_temp.csv" respectively. You can also set the parameter "save_file" to specify the file to save results. We use minibatch to reduce the memory occupation, you can modify it in the code. 

## 	Performance

```
                    |   Author's Code   |  GAMMAGL's Code   |
| Dataset | Metrics |  GRACE  |GRACE-POT|  GRACE  |GRACE-POT|
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|  Cora   |  Mi-F1  |   78.2  |   79.2  |   78.2  |   82.2  |
|  Cora   |  Ma-F1  |   76.8  |   77.8  |   77.1  |   81.3  |
|  PubMed |  Mi-F1  |   81.6  |   82.0  |   81.6  |   82.0  |
|  PubMed |  Ma-F1  |   81.7  |   82.4  |   80.5  |   80.1  |
|  Photo  |  Mi-F1  |   91.2  |   91.8  |   89.8  |   90.0  |
|  Photo  |  Ma-F1  |   89.2  |   90.0  |   88.5  |   87.9  |