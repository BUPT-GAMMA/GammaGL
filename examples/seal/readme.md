# SEAL

- Paper link: [https://arxiv.org/abs/1802.09691](https://arxiv.org/abs/1802.09691)


# Dataset Statics

| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

Results
-------

```bash
# k=2, available dataset: "cora", "citeseer", "pubmed"
TL_BACKEND='tensorflow' python -W ignore seal.py --data_dir /home/dhr/data --model gin --gpu_id 0
TL_BACKEND='torch' python -W ignore seal.py --data_dir /home/dhr/data --model gin --gpu_id 0 


| Dataset  | Paper | Our(pd)    | Our(tf)    | Our(torch) |
| -------- | ----- | ---------- | ---------- | ---------- |
| cora     |   |  | 93.28±0.58 | 83.52±1.18 |
| citeseer |  |  |  |  |
| pubmed   |  | |  |  |
