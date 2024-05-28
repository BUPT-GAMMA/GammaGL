@ -0,0 +1,40 @@
# Graph Contrastive Learning with Stable and Scalable

- Paper link: [https://proceedings.neurips.cc/paper_files/paper/2023/file/8e9a6582caa59fda0302349702965171-Paper-Conference.pdf](https://arxiv.org/abs/2201.11349)
- Author's code repo: [https://github.com/bdy9527/Sp2GCL](https://github.com/TaurusTaurus-Rui/DR-GST).

# Dataset Statics

| Dataset  | # Nodes | # Edges  | # Classes |
|----------|---------|----------|-----------|
| PubMed   | 19,717  | 88,648   | 3         |
| Wiki-CS  | 11,701  | 216,123  | 10        |
| Facebook | 22,470  | 342,004  | 4         |



Results
-------

```bash

TL_BACKEND="torch" python sp2gcl_trainer.py --dataset facebook
TL_BACKEND="torch" python sp2gcl_trainer.py --dataset wikics
TL_BACKEND="torch" python sp2gcl_trainer.py --dataset pubmed
```


# Dataset Statics

| Dataset  | Paper Code | Out(th)    |
|----------|------|------------|
| PubMed     | 82.3±0.3 | OOM        |
| Wiki-CS | 79.42±0.19 | 76.79 ± 0.61 |
| Facebook   | 90.43±0.13 | 85.35±0.26 |
