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
Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).


Results
-------
```bash
TL_BACKEND="tensorflow" python gcnii_trainer.py --dataset cora --lr 0.01 --num_layers 64 --alpha 0.1 --hidden_dim 64 --lambd 0.5 --drop_rate 0.3 --l2_coef 0.001
TL_BACKEND="tensorflow" python gcnii_trainer.py --dataset citeseer --lr 0.01 --num_layers 32 --alpha 0.1 --hidden_dim 256 --lambd 0.5 --drop_rate 0.3 --l2_coef 0.001
TL_BACKEND="tensorflow" python gcnii_trainer.py --dataset pubmed --lr 0.01 --num_layers 16 --alpha 0.1 --hidden_dim 256 --lambd 0.4 --drop_rate 0.3 --l2_coef 0.001
TL_BACKEND="paddle" python gcnii_trainer.py --dataset cora --lr 0.01 --num_layers 64 --alpha 0.1 --hidden_dim 64 --lambd 0.5 --drop_rate 0.3 --l2_coef 0.001
TL_BACKEND="paddle" python gcnii_trainer.py --dataset citeseer --lr 0.01 --num_layers 32 --alpha 0.1 --hidden_dim 256 --lambd 0.4 --drop_rate 0.4 --l2_coef 0.001
TL_BACKEND="paddle" python gcnii_trainer.py --dataset pubmed --lr 0.01 --num_layers 16 --alpha 0.1 --hidden_dim 256 --lambd 0.5 --drop_rate 0.7 --l2_coef 0.001
TL_BACKEND="torch" python gcnii_trainer.py --dataset cora --lr --lr 0.01 --num_layers 64 --alpha 0.1 --hidden_dim 64 --lambd 0.5 --drop_rate 0.3 --l2_coef 0.001
TL_BACKEND="torch" python gcnii_trainer.py --dataset citeseer --lr 0.01 --num_layers 64 --alpha 0.1 --hidden_dim 64 --lambd 0.6 --drop_rate 0.4 --l2_coef 0.001
TL_BACKEND="torch" python gcnii_trainer.py --dataset pubmed --lr 0.01 --num_layers 64 --alpha 0.1 --hidden_dim 64 --lambd 0.4 --drop_rate 0.6 --l2_coef 0.001
```
| Dataset  | Paper | Our(pd)      | Our(tf)      | Our(tf)    |
|----------|-------|--------------|--------------|------------|
| cora     | 85.5  | 83.12(±0.47) | 83.23(±0.76) | 83.1(±0.9) |
| pubmed   | 73.4  | 72.04(±0.91) | 71.9(±0.7)   | 71.4(±0.6) |
| citeseer | 80.3  | 80.36(±0.65) | 80.1(±0.5)   | 80.5(±0.3) |

> Notice that we do not use the same regularization method as the paper do, as TensorlayerX currently do not support it.