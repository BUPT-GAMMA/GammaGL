# Graph Contrastive Learning with Stable and Scalable

- Paper link: [https://arxiv.org/abs/2401.06595](https://arxiv.org/abs/2401.06595)
- Author's code repo: [https://github.com/q086/DyFSS](https://github.com/q086/DyFSS).

# Dataset Statics

| Dataset  | # Nodes | # Edges  | # Classes |
|----------|---------|----------|-----------|
| Cora     |  2,708  | 10,556   |    7      |
| Citeseer |  3,327  |  9,228   |    6      |
| Photo    |  7,650  | 238,162  |    8      |



Results
-------

```bash
TL_BACKEND="torch" python dyfss_trainer.py --dataset cora --lr 0.001 --pretrain_epochs 1000 --use_ckpt 0 --labels_epochs 500
TL_BACKEND="torch" python dyfss_trainer.py --dataset citeseer --lr 0.001 --pretrain_epochs 1000 --use_ckpt 0 --labels_epochs 500
TL_BACKEND="torch" python dyfss_trainer.py --dataset photo --lr 0.001 --pretrain_epochs 1000 --use_ckpt 0 --labels_epochs 500
```


# Dataset Statics

| Dataset  | Paper Code | Out(th)    |
| -------- | ---------- | ---------- |
| Cora     | 60.7±0.3   | 61.7±0.76  |
| Citeseer | 56.3±0.19  | 66.0±0.20  |
| Photo    | 79.4±0.13  | 79.2±0.34  |

