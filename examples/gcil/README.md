# Graph Contrastive Invariant Learning from the Causal Perspective (GCIL)

- Paper link: [https://arxiv.org/pdf/2401.12564v2](https://arxiv.org/pdf/2401.12564v2)
- Author's code repo: [https://github.com/BUPT-GAMMA/GCIL](https://github.com/BUPT-GAMMA/GCIL). Note that the original code is 
  implemented with Tensorflow for the paper. 

# Dataset Statics

| Dataset | # Nodes | # Edges | # Classes |
| ------- | ------- | ------- | --------- |
| Cora    | 2,708   | 10,556  | 7         |
| Pubmed  | 19,717  | 88,651  | 3         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

Results
-------

```bash
# available dataset: "cora", "pubmed"
TL_BACKEND="torch" python gcil_trainer.py cora
TL_BACKEND="torch" python gcil_trainer.py pubmed 
```

Ma-F1:
| Dataset | Paper    | Our(th)    |
| ------- | -------- | ---------- |
| cora    | 83.8±0.5 | 45.19±0.22 |
| pubmed  | 81.5±0.5 | 46.30±0.02 |

Mi-F1
| Dataset | Paper    | Our(th)    |
| ------- | -------- | ---------- |
| cora    | 84.4±0.7 | 49.71±0.22 |
| pubmed  | 81.6±0.7 | 53.77±0.01 |