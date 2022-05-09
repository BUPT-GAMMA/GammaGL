# Simple and Deep Graph Convolutional Networks (GCNII)

- Paper link: [https://arxiv.org/abs/2007.02133](https://arxiv.org/abs/2007.02133)
- Author's code repo: [https://github.com/chennnM/GCNII](https://github.com/chennnM/GCNII). 
> Note that our implementation is little different with the author's in the optimizer.  The author applied different weight decay coefficient on learnable paramenters, while TensorLayerX has not support this feature.

Dataset Statics
-------
| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

How to run
----------

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python gcnii_train.py --dataset cora
```
> For details settings, please refer to [here](https://github.com/BUPT-GAMMA/GammaGL/tree/main/examples/gcn#how-to-run).

Results
-------
```bash
python gcnii_trainer.py --dataset cora --lr 0.01 --num_layers 64 --alpha 0.1 --hidden_dim 64 --lambd 0.5 --drop_rate 0.3 --l2_coef 0.001
python gcnii_trainer.py --dataset pubmed --lr 0.01 --num_layers 32 --alpha 0.1 --hidden_dim 256 --lambd 0.5 --drop_rate 0.3 --l2_coef 0.001
python gcnii_trainer.py --dataset citeseer --lr 0.01 --num_layers 16 --alpha 0.1 --hidden_dim 256 --lambd 0.4 --drop_rate 0.3 --l2_coef 0.001
```
| Dataset  | Paper | Our(pd)      | Our(tf)      |
|----------|-------|--------------|--------------|
| cora     | 85.5  | 83.12(±0.47) | 83.23(±0.76) |
| pubmed   | 73.4  | 72.04(±0.91) | 71.9(±0.7)   |
| citeseer | 80.3  | 80.36(±0.65) | 80.1(±0.5)   |

