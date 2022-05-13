# Frequency Adaptation Graph Convolutional Networks (FAGCN)

- Paper link: [https://arxiv.org/abs/2101.00797](https://arxiv.org/abs/2101.00797)
- Author's code repo: [https://github.com/bdy9527/FAGCN](https://github.com/bdy9527/FAGCN)

# Dataset Statics
| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

# How to Run
Run with following commands:
(available dataset: "cora", "citeseer", "pubmed")  
(backend is  `tensorflow` and gpu id is 0 by DEFAULT)  
> Note: TensorFlow will take up all GPU left memory by default.

```bash
python fagcn_trainer.py --dataset cora --lr 0.01
```
If you want to use specific `backend` and `GPU`, just set environment variable like:
```bash
CUDA_VISIBLE_DEVICES="1" TL_BACKEND="tensorflow" python fagcn_trainer.py
```
> Note: Set `CUDA_VISIBLE_DEVICES=""` if you want to execute script in CPU.


Results
-------
```bash
TL_BACKEND="tensorflow" python fagcn_trainer.py --dataset cora --lr 0.01 --l2_coef 0.0005 --drop_rate 0.4 --hidden_dim 16 --eps 0.3 --num_layers 5
TL_BACKEND="tensorflow" python fagcn_trainer.py --dataset citeseer --lr 0.01 --l2_coef 0.0005 --drop_rate 0.4 --hidden_dim 16 --eps 0.4 --num_layers 4
TL_BACKEND="tensorflow" python fagcn_trainer.py --dataset pubmed --lr 0.01 --l2_coef 0.001 --drop_rate 0.4 --hidden_dim 16 --eps 0.1 --num_layers 6
```
| Dataset  | Paper    | Our(pd)  |
|----------|----------|----------|
| cora     | 84.1±0.5 | 83.1±0.4 |
| citeseer | 72.7±0.8 | 68.3±0.8 |
| pubmed | 79.4±0.3 | 79.1±0.2 |

