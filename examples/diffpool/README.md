# Differentiable Pooling

- Paper link: [https://arxiv.org/abs/1806.08804]( https://arxiv.org/abs/1806.08804)
- Author's code repo: [https://github.com/RexYing/diffpool](https://github.com/RexYing/diffpool). Note that the original code is 
  implemented with Pytorch for the paper. 

# Dataset Statics

| Dataset | # Graphs | # Edges | # Avg of graph size |
|---------|----------|---------|---------------------|
| syn1v2  | 1000     | 181067  | 49.23               |
| D&D     | 1168     | 789819  | 268.70              |
| ENZYMES | 600      | 37282   | 32.46               |

Results
-------

```bash
python diffpool_trainer.py --dataset syn1v2 --num_pool 1 --epochs 50
python diffpool_trainer.py --bmname DD
```

| Dataset | Paper | Our(pytorch) | Our(tf) |
|---------|-------|--------------|---------|
| syn1v2  | -     | 0.80         |         |
| D&D     | 81.15 | 79.31        |         |
| ENZYMES | 64.23 |              |         |