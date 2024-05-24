# Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning

- Paper link: [https://arxiv.org/abs/2105.09111](https://arxiv.org/abs/2105.09111)
- Author's code repo: [https://github.com/liun-online/HeCo](https://github.com/liun-online/HeCo)

# Dataset Statics
| Dataset  | # Nodes_paper | # Nodes_author | # Nodes_subject |
|----------|---------------|----------------|-----------------|
| ACM      | 4019          | 7167           | 60              |

Refer to [ACM](https://github.com/AndyJZhao/NSHE/tree/master/data/acm).

Results For ACM
-------
- Ma-F1

| number of train_labels  | Paper    | Our(tf)  | Our(pd)  | Our(torch) |
|-------------------------|----------|----------|----------|------------|
|    20                   | 88.56±0.8| 82.5±0.4 | 82.7±0.4 | 82.6±0.3   |
|    40                   | 87.61±0.5| 85.1±0.3 | 85.4±0.1 | 85.6±0.2   |
|    60                   | 89.04±0.5| 86.4±0.4 | 86.3±0.4 | 86.4±0.3   |

- Mi-F1

| number of train_labels  | Paper    | Our(tf)  | Our(pd)  | Our(torch) |
|-------------------------|----------|----------|----------|------------|
|    20                   | 88.13±0.8| 81.1±0.4 | 81.8±0.8 | 82.4±0.4   |
|    40                   | 87.45±0.5| 83.4±0.3 | 85.43±0.1| 85.4±0.6   |
|    60                   | 88.71±0.5| 84.4±0.4 | 85.2±0.5 | 85.4±0.6   |


- AUC

| number of train_labels  | Paper    | Our(tf)  | Our(pd)  | Our(torch) |
|-------------------------|----------|----------|----------|------------|
|    20                   | 96.49±0.3| 93.8±0.4 | 94.1±0.4 | 94.3±0.3   |
|    40                   | 96.4±0.4 | 94.4±0.3 | 95.1±0.2 | 95.4±0.3   |
|    60                   | 96.55±0.3| 94.8±0.4 | 95.4±0.4 | 95.7±0.4   |

For TensorFlow runs more slowly than paddlepaddle and pytorch, thus pd and torch are more recommended.
