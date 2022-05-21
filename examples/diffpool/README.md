# Graph Convolutional Networks (GCN)

- Paper link: [ https://arxiv.org/abs/1806.08804]( https://arxiv.org/abs/1806.08804)
- Author's code repo: [https://github.com/RexYing/diffpool](https://github.com/RexYing/diffpool). Note that the original code is 
  implemented with Pytorch for the paper. 

# Dataset Statics

| Dataset | # Graphs | # Edges  | # Avg of graph size |
|---------|----------|----------|---------------------|
| syn1v2  |  1000    | 181067   | 49.23               |
| D&D     |          |          |                     |
| ENZYMES |          |          |                     |

Results
-------

```bash
# available dataset: "DD", "ENZYMES"
python diffpool_trainer.py --dataset syn1v2 --num_pool 1 --epochs 50
#python diffpool_trainer.py --dataset ENZYMES --num_pool 1 --epochs 1000
#python diffpool_trainer.py --dataset DD --num_pool 1  --batch-size 10
```

| Dataset | Paper | Our(pytorch) | Our(tf) |
|---------|-------|--------------|---------|
| syn1v2  |       | 0.800002     |         |
| D&D     | 81.15 |              |         |
| ENZYMES | 64.23 |              |         |