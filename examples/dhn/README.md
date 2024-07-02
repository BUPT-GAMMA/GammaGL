# Distance encoding based Heterogeneous graph neural Network (DHN)
- Paper link: [https://ieeexplore.ieee.org/document/10209229](https://ieeexplore.ieee.org/document/10209229)
- Author's code repo: [https://github.com/BUPT-GAMMA/HDE](https://github.com/BUPT-GAMMA/HDE)

## Dataset Statics
| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Cora     | 4435    | 7071    | 2         |

## Results
```bash
TL_BACKEND="torch" python dhn_trainer.py --test_ratio 0.3 --one_hot True --k_hop 2 --num_neighbor 5 --batch_size 32 --lr 0.001 --n_epoch 100 --drop_rate 0.01 --dataset 'acm'
```

| Dataset  | Paper(AUC) | Our(th)(AUC)     |
| -------- | ----- | ----------- | 
| acm      | 95.07 |  95.54±0.18 | 