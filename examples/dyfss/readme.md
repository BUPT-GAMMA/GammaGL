# Every Node is Different: Dynamically Fusing Self-Supervised Tasks for Attributed Graph Clustering

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
cd GammaGL/examples/dyfss
TL_BACKEND="torch" python dyfss_trainer.py --dataset cora --pretrain_epochs 1000 --use_ckpt 0 --labels_epochs 500 --top_k 5
TL_BACKEND="torch" python dyfss_trainer.py --dataset citeseer --pretrain_epochs 1000 --use_ckpt 0 --labels_epochs 500 --top_k 5
TL_BACKEND="torch" python dyfss_trainer.py --dataset photo --pretrain_epochs 1000 --use_ckpt 0 --labels_epochs 500 --top_k 5
```


# Dataset Statics

| Dataset  | Paper Code | Out(th)    |
| -------- | ---------- | ---------- |
| Cora     | 60.7±0.70  | 61.7±2.74  |
| Citeseer | 56.3±0.62  | 66.0±2.76  |
| Photo    | 79.4±0.47  | 79.2±1.16  |

