Graph Attention Networks v2 (GATv2)
============

- Paper link: [How Attentive are Graph Attention Networks?](https://arxiv.org/pdf/2105.14491.pdf)
- Author's code repo: [https://github.com/tech-srl/how_attentive_are_gats](https://github.com/tech-srl/how_attentive_are_gats).
- Annotated implemetnation: [https://nn.labml.ai/graphs/gatv2/index.html]

Dataset Statics
-------
| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

Results
-------
```bash
TL_BACKEND="paddle" python gatv2_trainer.py --dataset cora --lr 0.01 --l2_coef 0.01 --drop_rate 0.7
TL_BACKEND="paddle" python gatv2_trainer.py --dataset citeseer --lr 0.0005 --l2_coef 0.01 --drop_rate 0.3
TL_BACKEND="paddle" python gatv2_trainer.py --dataset pubmed --lr 0.01 --l2_coef 0.001 --drop_rate 0.4
```
| Dataset  | Our(pd)     | Our(tf)     |
| -------- | ----------- | ----------- |
| cora     | 82.45(±0.34) | 81.78(±0.29) |
| pubmed   | 70.9(±1.28)  | 69.9(±0.23)  |
| citeseer | 78.46(±0.19) | 77.49(±0.08) |
