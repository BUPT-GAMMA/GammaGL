Edge-augmented Graph Transformer (EGT)
============

- Paper link: [https://arxiv.org/abs/2108.03348](https://arxiv.org/abs/2108.03348)
- Author's code repo
  - in Tensorflow:
  [https://github.com/shamim-hussain/egt.git](https://github.com/shamim-hussain/egt.git).
  - in Pytorch:
  [https://github.com/shamim-hussain/egt_pytorch.git](https://github.com/shamim-hussain/egt_pytorch.git).

Dataset Statics
-------

| Dataset  | # Graphs  | # Nodes    | # Edges     |
| -------- | --------- | ---------- | ----------- |
| PCQM4Mv2 | 3,746,620 | 52,970,652 | 109,093,626 |


Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

Results
-------

```bash

TL_BACKEND="torch" python examples/egt/egt_trainer.py examples/egt/config.yaml
TL_BACKEND="tensorflow" python examples/egt/egt_trainer.py examples/egt/config.yaml
```

| Dataset  | Paper      | Our(torch)      | Our(tf)   |
| -------- | ---------- | ------------ | ------------ |
| PCQM4Mv2 | 0.0869     | 0.0873(±0.007) | 0.0884(±0.009) |

