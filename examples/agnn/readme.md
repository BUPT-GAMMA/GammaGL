# Attention-based Graph Neural Network(AGNN)

- Paper link: [http://arxiv.org/abs/1803.03735](http://arxiv.org/abs/1803.03735):

- Popular pytorch implementation:
  [https://github.com/dawnranger/pytorch-AGNN](https://github.com/dawnranger/pytorch-AGNN)

## Dataset Statics
| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

## How to run

Run with following:

```bash
python agnn_trainer.py
```
> For details settings, please refer to [here](https://github.com/BUPT-GAMMA/GammaGL/tree/main/examples/gcn#how-to-run).

## Results
```bash
TL_BACKEND="tensorflow" python agnn_trainer.py --dataset cora --lr 0.01 --l2_coef 0.002 --drop_rate 0.7
TL_BACKEND="tensorflow" python agnn_trainer.py --dataset citeseet --lr 0.005 --l2_coef 0.006 --drop_rate 0.5
```
| Dataset  | Paper | Our(tf)     | Our(pd) | Our(torch) |
| -------- | ----- | ----------- | ------- | ---------- |
| cora     | 83.1  | 83.28(±0.64) |         |            |
| citeseer | 71.7  | 71.7(±0.53)  |         |            |
| pubmed   | 79.9  | 79.02(±0.82) |         |            |

