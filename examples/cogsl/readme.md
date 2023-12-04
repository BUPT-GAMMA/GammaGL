# Compact Graph Structure Learning via Mutual Information Compression (CoGSL)

- Paper link: [https://arxiv.org/abs/2201.05540](https://arxiv.org/abs/2201.05540)
- Author's code repo: [https://github.com/liun-online/CoGSL](https://github.com/liun-online/CoGSL).

# Dataset Statics

| Dataset      | # Nodes | # Edges | # Classes |
|--------------|---------|---------|-----------|
| Polblogs     | 1222    | 33428   | 2         |
| Citeseer     | 3,327   | 9,228   | 6         |
| Wiki-CS      | 11701   | 291039  | 10        |
| MS  Academic | 18333   | 163788  | 15        |


Results
-------

```bash
# available dataset: "polblogs", "citeseer", "wikics", "ms"
TL_BACKEND="paddle" python cogsl_trainer.py polblogs
TL_BACKEND="paddle" python cogsl_trainer.py citeseer
TL_BACKEND="paddle" python cogsl_trainer.py wikics
TL_BACKEND="paddle" python cogsl_trainer.py ms
TL_BACKEND="torch" python cogsl_trainer.py polblogs
TL_BACKEND="torch" python cogsl_trainer.py citeseer
TL_BACKEND="torch" python cogsl_trainer.py wikics
TL_BACKEND="torch" python cogsl_trainer.py ms


```

| Dataset       | Metric                        | Paper                              | Our(pd)                            | Our(th)                            |
|---------------|-------------------------------|------------------------------------|------------------------------------|------------------------------------|
| polblogs      | F1-macro<br/>F1-micro<br/>AUC | 95.5±0.1<br/>95.5±0.1<br/>98.3±0.1 | 96.0±0.2<br/>96.0±0.2<br/>98.7±0.0 | 95.2±0.4<br/>95.2±0.4<br/>98.2±0.1 |
| citeseer      | F1-macro<br/>F1-micro<br/>AUC | 70.2±0.6<br/>73.4±0.8<br/>91.4±0.5 | 69.0±0.3<br/>72.4±0.5<br/>91.1±0.2 | 64.3±0.5<br/>67.3±0.5<br/>88.0±0.3 |
| wikics        | F1-macro<br/>F1-micro<br/>AUC | 72.3±0.6<br/>75.0±0.3<br/>96.4±0.2 | 71.3±0.4<br/>74.2±0.3<br/>95.3±0.3 | 65.1±0.5<br/>69.0±0.8<br/>93.5±0.4 |
| ms            | F1-macro<br/>F1-micro<br/>AUC | 90.5±0.4<br/>92.4±0.5<br/>99.4±0.1 | 88.1±0.0<br/>90.1±0.0<br/>98.9±0.0 | 77.8±0.0<br/>82.9±0.0<br/>98.0±0.0 |
