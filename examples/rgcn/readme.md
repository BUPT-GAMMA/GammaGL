# Relational Graph Convolutional Network
Paper: Modeling Relational Data with Graph Convolutional Networks
Author's code for entity classification: [https://github.com/tkipf/relational-gcn](https://github.com/tkipf/relational-gcn)
Author's code for link prediction: [https://github.com/MichSchli/RelationPrediction](https://github.com/MichSchli/RelationPrediction)

# Dataset Statics
| Dataset | #Nodes    | #Edges     | #Relations | #Labeled |
|---------|-----------|------------|------------|----------|
| AIFB    | 8,285     | 58,086     | 90         | 176      |
| MUTAG   | 23,644    | 148,454    | 46         | 340      |
| BGS     | 333,845   | 1,832,398  | 206        | 146      |
| AM      | 1,666,764 | 11,976,642 | 266        | 1000     |

Results
-------

```bash
TL_BACKEND="paddle" python rgcn_trainer.py --dataset aifb 
TL_BACKEND="paddle" python rgcn_trainer.py --dataset mutag --lr 0.001 --l2_coef 5e-2
TL_BACKEND="paddle" python rgcn_trainer.py --dataset bgs --lr 0.001 --l2_coef 1e-2
```

| Dataset | Paper | Our(th)    | Our(tf)   |
|---------|-------|------------|-----------|
| AIFB    | 95.83 | 93.8(±2.0) | 94.44(±0) |
| MUTAG   | 73.23 | 82.3(±1.8) |           |
| BGS     | 83.10 | 74.1(±1.7) |           |
| AM      | 89.29 |            |           |
