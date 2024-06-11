Heterogeneous Edge-Enhanced Graph Attention Network(HEAT)
============

- Paper link: [https://arxiv.org/abs/2106.07161](https://arxiv.org/abs/2106.07161)
- Author's code repo (in PyTorch):
  [https://github.com/Xiaoyu006/MATP-with-HEAT](https://github.com/Xiaoyu006/MATP-with-HEAT).

Dataset Statics
-------

| Dataset      | # Number of Graphs | # Type of Nodes | # Type of Edges |
|--------------|--------------------|-----------------|-----------|
| NGSIM US-101 | 1201               | 2               | 2         |

Refer to [NGSIM US-101 Datasets](https://github.com/gjy1221/NGSIM-US-101).

Results
-------

```bash
TL_BACKEND="torch" python heat_trainer.py --data_path ../data --result_path ../result
```

| Time(sec) | Paper  | Our(torch) |
|-----------|--------|------------|
| 1         | 0.6067 | 0.6940     |
| 2         | 0.8556 | 0.7349     |
| 3         | 1.0469 | 1.0621     |
| 4         | 1.3216 | 1.2739     |
| 5         | 1.8894 | 1.7562     |