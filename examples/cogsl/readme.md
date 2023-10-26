# Compact Graph Structure Learning via Mutual Information Compression (CoGSL)

- Paper link: [https://arxiv.org/abs/2201.05540](https://arxiv.org/abs/2201.05540)
- Author's code repo: [https://github.com/liun-online/CoGSL](https://github.com/liun-online/CoGSL).

# Dataset Statics

| Dataset      | # Nodes | # Edges | # Classes |
|--------------|---------|---------|-----------|
| Wine         | 178     | 3560    | 3         |
| Cancer       | 569     | 22760   | 2         |
| Digits       | 1797    | 43128   | 10        |
| Polblogs     | 1222    | 33428   | 2         |
| Citeseer     | 3,327   | 9,228   | 6         |
| Wiki-CS      | 11701   | 291039  | 10        |
| MS  Academic | 18333   | 163788  | 15        |


Results
-------

```bash
# available dataset: "wine", "breast_cancer", "digits", "polblogs", "citeseer", "wikics", "ms"
TL_BACKEND="paddle" python cogsl_trainer.py wine
TL_BACKEND="paddle" python cogsl_trainer.py breast_cancer
TL_BACKEND="paddle" python cogsl_trainer.py digits
TL_BACKEND="paddle" python cogsl_trainer.py polblogs
TL_BACKEND="paddle" python cogsl_trainer.py citeseer
TL_BACKEND="paddle" python cogsl_trainer.py wikics
TL_BACKEND="torch" python cogsl_trainer.py wine
TL_BACKEND="torch" python cogsl_trainer.py breast_cancer
TL_BACKEND="torch" python cogsl_trainer.py digits
TL_BACKEND="torch" python cogsl_trainer.py polblogs
TL_BACKEND="torch" python cogsl_trainer.py citeseer
TL_BACKEND="torch" python cogsl_trainer.py wikics


```

| Dataset       | Metric                        | Paper                              | Our(pd)                            | Our(tf)    | Our(th)                            | Our(ms)    |
|---------------|-------------------------------|------------------------------------|------------------------------------|------------|------------------------------------|------------|
| wine          | F1-macro<br/>F1-micro<br/>AUC | 97.9±0.3<br/>97.8±0.3<br/>99.7±0.1 | 97.5±0.3<br/>97.4±0.3<br/>99.8±0.1 |  | 97.9±0.7<br/>97.8±0.7<br/>99.7±0.3 |  |
| breast_cancer | F1-macro<br/>F1-micro<br/>AUC | 94.6±0.3<br/>95.0±0.3<br/>98.5±0.1 | 94.2±0.5<br/>94.6±0.5<br/>98.4±0.3 |  | 94.0±0.5<br/>94.4±0.5<br/>98.1±0.4 |  |
| digits        | F1-macro<br/>F1-micro<br/>AUC | 93.3±0.3<br/>93.3±0.3<br/>99.6±0.0 | 92.5±0.3<br/>92.6±0.3<br/>99.5±0.1 |  | 92.0±0.3<br/>92.1±0.3<br/>99.3±0.1 |  |
| polblogs      | F1-macro<br/>F1-micro<br/>AUC | 95.5±0.1<br/>95.5±0.1<br/>98.3±0.1 | 96.0±0.2<br/>96.0±0.2<br/>98.7±0.0 |  | 95.2±0.4<br/>95.2±0.4<br/>98.2±0.1 |  |
| citeseer      | F1-macro<br/>F1-micro<br/>AUC | 70.2±0.6<br/>73.4±0.8<br/>91.4±0.5 |                                    |  |                                    |  |
| wikics        | F1-macro<br/>F1-micro<br/>AUC | 72.3±0.6<br/>75.0±0.3<br/>96.4±0.2 |                                    |  |                                    |  |
| ms            | F1-macro<br/>F1-micro<br/>AUC | 90.5±0.4<br/>92.4±0.5<br/>99.4±0.1 |                                    |  |                                    |  |
