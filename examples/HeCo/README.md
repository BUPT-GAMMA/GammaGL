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
|    20                   | 88.56±0.8| 79.1±0.4 | 81.7±0.4 | 81.6±0.3   |
|    40                   | 87.61±0.5| 83.4±0.3 | 85.4±0.1 | 85.6±0.2   |
|    60                   | 89.04±0.5| 81.4±0.4 | 83.4±0.4 | 83.3±0.3   |

- Mi-F1

| number of train_labels  | Paper    | Our(tf)  | Our(pd)  | Our(torch) |
|-------------------------|----------|----------|----------|------------|
|    20                   | 88.13±0.8| 79.1±0.4 | 80.44±0.8| 80.4±0.7   |
|    40                   | 87.45±0.5| 83.4±0.3 | 85.43±0.1| 85.4±0.2   |
|    60                   | 88.71±0.5| 79.4±0.4 | 82.2±0.5 | 82.4±0.6   |


- AUC

| number of train_labels  | Paper    | Our(tf)  | Our(pd)  | Our(torch) |
|-------------------------|----------|----------|----------|------------|
|    20                   | 96.49±0.3| 89.8±0.4 | 92.8±0.4 | 92.8±0.3   |
|    40                   | 96.4±0.4 | 92.4±0.3 | 95.2±0.2 | 95.4±0.3   |
|    60                   | 96.55±0.3| 89.4±0.4 | 93.6±0.4 | 93.7±0.3   |

For TensorFlow runs more slowly than paddlepaddle and pytorch, thus pd and torch are more recommended.