# Relational Graph Convolutional Network
- Paper: Modeling Relational Data with Graph Convolutional Networks

- Author's code for entity classification: [https://github.com/tkipf/relational-gcn](https://github.com/tkipf/relational-gcn)
- Author's code for link prediction: [https://github.com/MichSchli/RelationPrediction](https://github.com/MichSchli/RelationPrediction)

# Dataset Statics
| Dataset | #Nodes    | #Edges     | #Relations | #Labeled |
| ------- | --------- | ---------- | ---------- | -------- |
| AliRCD   | 13806619 | 157814864  |     14     |    2     |


Results
-------

```bash
TL_BACKEND="torch" python rgcn_trainer.py

TL_BACKEND="tensorflow" python rgcn_trainer.py 

```

| Dataset |  Our(th)      | Our(tf)      | Our(pd)     |
|---------|--------------|--------------|--------------|
| AliRCD  | 92.50(±0.22) | 92.50(±0.22) |    ----      |
