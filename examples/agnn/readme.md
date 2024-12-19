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
TL_BACKEND=tensorflow python agnn_trainer.py --dataset pubmed --n_att_layers 4 --lr 0.008 --l2_coef 0.005 --drop_rate 0.5
TL_BACKEND=torch python agnn_trainer.py --dataset cora --lr 0.01 --l2_coef 0.005 --drop_rate 0.7
TL_BACKEND=torch python agnn_trainer.py --dataset citeseer --lr 0.001 --l2_coef 0.02 --drop_rate 0.5
TL_BACKEND=torch python agnn_trainer.py --dataset pubmed --n_att_layers 4 --lr 0.01 --l2_coef 0.002 --drop_rate 0.7
TL_BACKEND=paddle python agnn_trainer.py --dataset cora --lr 0.02 --l2_coef 0.001 --drop_rate 0.5
TL_BACKEND=paddle python agnn_trainer.py --dataset citeseer --lr 0.005 --l2_coef 0.01 --drop_rate 0.7
TL_BACKEND=paddle python agnn_trainer.py --dataset pubmed --n_att_layers 4 --lr 0.02 --l2_coef 0.002 --drop_rate 0.6
```
| Dataset  | Paper | Our(tf)     | Our(pd) | Our(torch) |
| -------- | ----- | ----------- | ------- | ---------- |
| cora     | 83.1  | 83.28(±0.64) | 83.48(±0.35) | 83.0(±0.65) |
| citeseer | 71.7  | 71.7(±0.53)  | 73.24(±0.71) | 72.52(±1.13) |
| pubmed   | 79.9  | 79.02(±0.82) | 78.94(±0.43) | 79.10(±0.20) |

