# Composition-based Multi-Relational Graph Convolutional Networks

- Paper: Composition-based Multi-Relational Graph Convolutional Networks
- Author's code for link prediction: [https://github.com/malllabiisc/CompGCN](https://github.com/MichSchli/RelationPrediction)

# Dataset Statics
| Dataset | #Nodes    | #Edges     | #Relations | #Labeled |
| ------- | --------- | ---------- | ---------- | -------- |
| AIFB    | 8,285     | 58,086     | 90         | 176      |
| MUTAG   | 23,644    | 148,454    | 46         | 340      |
| BGS     | 333,845   | 1,832,398  | 206        | 146      |

Results
-------

```bash
TL_BACKEND="pytorch" python comgcn_trainer.py --dataset aifb  --l2 5e-5 --op sub
TL_BACKEND="pytorch" python comgcn_trainer.py --dataset mutag --lr 0.015 --l2_coef 5e-2 --op sub
TL_BACKEND="pytorch" python comgcn_trainer.py --dataset bgs --lr 0.0001 --l2_coef 5e-2
TL_BACKEND="paddle" python comgcn_trainer.py --dataset aifb  --l2 5e-5 --op sub
TL_BACKEND="paddle" python comgcn_trainer.py --dataset mutag --lr 0.015 --l2_coef 5e-2 --op sub
TL_BACKEND="paddle" python comgcn_trainer.py --dataset bgs --lr 0.0001 --l2_coef 5e-2
```

| Dataset | Paper      | Our(th)      | Our(pd)      |
| ------- | ---------- | ------------ | ------------ |
| AIFB    | /          | 88.89(±2.78) | 87.22(±1.36) |
| MUTAG   | 85.3(±1.2) | 82.0(±2.66)  | 81.47(±2.73) |
| BGS     | /          | 77.93(±4.14) | 75.17(±4.02) |