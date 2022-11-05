# Composition-based Multi-Relational Graph Convolutional Networks

- Paper: Composition-based Multi-Relational Graph Convolutional Networks
- Author's code for link prediction: [https://github.com/malllabiisc/CompGCN](https://github.com/MichSchli/RelationPrediction)

# Dataset Statics
| Dataset | #Nodes    | #Edges     | #Relations | #Labeled |
| ------- | --------- | ---------- | ---------- | -------- |
| AIFB    | 8,285     | 58,086     | 90         | 176      |
| MUTAG   | 23,644    | 148,454    | 46         | 340      |
| BGS     | 333,845   | 1,832,398  | 206        | 146      |
| AM      | 1,666,764 | 11,976,642 | 266        | 1000     |

Results
-------

```bash
TL_BACKEND="pytorch" python rgcn_trainer.py --dataset aifb  --l2 5e-5 --op sub
TL_BACKEND="pytorch" python rgcn_trainer.py --dataset mutag --lr 0.015 --l2_coef 5e-2 --op sub
TL_BACKEND="pytorch" python rgcn_trainer.py --dataset bgs --lr 0.0001 --l2_coef 5e-2
```

| Dataset | Our(th)      | Our(tf) | Our(pd) |
| ------- | ------------ | ------- | ------- |
| AIFB    | 88.89(±2.78) | /       | /       |
| MUTAG   | 82.0(±2.66)  | /       | /       |
| BGS     | 68.97        | /       | /       |
| AM      |              |         |         |