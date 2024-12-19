# Relational Graph Convolutional Network
- Paper: Modeling Relational Data with Graph Convolutional Networks

- Author's code for entity classification: [https://github.com/tkipf/relational-gcn](https://github.com/tkipf/relational-gcn)
- Author's code for link prediction: [https://github.com/MichSchli/RelationPrediction](https://github.com/MichSchli/RelationPrediction)

# Dataset Statics
| Dataset | #Nodes    | #Edges     | #Relations | #Labeled |
| ------- | --------- | ---------- | ---------- | -------- |
| AIFB    | 8,285     | 58,086     | 90         | 176      |
| MUTAG   | 23,644    | 148,454    | 46         | 340      |
| BGS     | 333,845   | 1,832,398  | 206        | 146      |
| AM      | 1,666,764 | 11,976,642 | 266        | 1000     |



| Dateset | #Nodes | #Node Types | #Edges  | #Relations | Target | #Classes |
| ------- | ------ | ----------- | ------- | ---------- | ------ | -------- |
| IMDB    | 11,616 | 3           | 102,804 | 4          | movie  | 3        |



Results
-------

```bash
TL_BACKEND="torch" python rgcn_trainer.py --dataset aifb  --l2 5e-5
TL_BACKEND="torch" python rgcn_trainer.py --dataset mutag --l2_coef 5e-2
TL_BACKEND="torch" python rgcn_trainer.py --dataset bgs --lr 0.0001 --l2_coef 5e-2
TL_BACKEND="torch" python rgcn_trainer.py --dataset imdb

TL_BACKEND="tensorflow" python rgcn_trainer.py --dataset aifb 
TL_BACKEND="tensorflow" python rgcn_trainer.py --dataset mutag --l2_coef 5e-2
TL_BACKEND="tensorflow" python rgcn_trainer.py --dataset bgs --l2_coef 5e-2
TL_BACKEND="tensorflow" python rgcn_trainer.py --dataset imdb

TL_BACKEND="paddle" python rgcn_trainer.py --dataset aifb 
TL_BACKEND="paddle" python rgcn_trainer.py --dataset mutag --l2_coef 5e-2
TL_BACKEND="paddle" python rgcn_trainer.py --dataset bgs --l2_coef 5e-2
TL_BACKEND="paddle" python rgcn_trainer.py --dataset imdb
```

| Dataset | Paper | Our(th)      | Our(tf)      | Our(pd)      |
| ------- | ----- | ------------ | ------------ | ------------ |
| AIFB    | 95.83 | 96.11(±1.52) | 94.17(±2.05) | 95.56(±2.3)  |
| MUTAG   | 73.23 | 85.0(±0.66)  | 85.29(±1.20) | 85.00(±1.9)  |
| BGS     | 83.10 | 74.1(±1.7)   | 73.79(±1.9)  | 73.56(±3.8)  |
| AM      | 89.29 |              |              |              |
| IMDB    |       | 48.54(±0.62) | 48.30(±1.20) | 48.44(±1.10) |