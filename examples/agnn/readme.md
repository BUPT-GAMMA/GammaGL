# Attention-based Graph Neural Network(AGNN)

- Paper link: [http://arxiv.org/abs/1803.03735](http://arxiv.org/abs/1803.03735):

- Popular pytorch implementation:
  [https://github.com/dawnranger/pytorch-AGNN](https://github.com/dawnranger/pytorch-AGNN)

## Dataset Statics

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

## Results
```bash
TL_BACKEND=tensorflow python agnn_trainer.py --dataset cora --lr 0.01 --l2_coef 0.002 --drop_rate 0.7
TL_BACKEND=tensorflow python agnn_trainer.py --dataset citeseer --lr 0.005 --l2_coef 0.006 --drop_rate 0.5
```
| Dataset  | Paper | Our(tf)     | Our(pd) | Our(torch) |
| -------- | ----- | ----------- | ------- | ---------- |
| cora     | 83.1  | 83.28(±0.64) |         |            |
| citeseer | 71.7  | 71.7(±0.53)  |         |            |
| pubmed   | 79.9  | 79.02(±0.82) |         |            |

