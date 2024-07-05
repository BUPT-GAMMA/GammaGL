# Graph-less Neural Networks (GLNN)

- Paper link: [https://arxiv.org/pdf/2205.03811](https://arxiv.org/pdf/2205.03811)
- Author's code repo: [https://anonymous.4open.science/r/DF-GNNs-EC75](https://anonymous.4open.science/r/DF-GNNs-EC75)

# Dataset Statics
| Dataset  | # Graphs | # Nodes | # Edges | # Features | # Classes |
| -------- | -------- | ------- | ------- | ---------- | --------- |
| MUTAG    | 188      | ~17.9   | ~39.6   | 7          | 2         |

Refer to [TUDataset](https://gammagl.readthedocs.io/en/latest/generated/gammagl.datasets.TUDataset.html).

# Results

```bash
TL_BACKEND="torch" python train_teacher.py --dataset MUTAG 
TL_BACKEND="torch" python train_student.py --dataset MUTAG --student gcn
TL_BACKEND="torch" python train_student.py --dataset MUTAG --student gin
```

| Dataset    | Student Model | Paper      | Our(th)    |
| ---------  | ------------- | ---------- | ---------- |
| MUTAG      | gcn           | 76.2%      | 88.2%      |
| MUTAG      | gin           | 90.8%      | 88.2%      |
| MUTAG      | gat           | 79.5%      | 88.2%      |
| MUTAG      | graphsage     | 79.1%      | 88.2%      |