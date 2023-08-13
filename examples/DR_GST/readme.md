# Distribution Recovered Graph Self-Training framework (DR-GST)

- Paper link: [https://arxiv.org/abs/2201.11349](https://arxiv.org/abs/2201.11349)
- Author's code repo: [https://github.com/TaurusTaurus-Rui/DR-GST](https://github.com/TaurusTaurus-Rui/DR-GST). Note that the original code is implemented with Pytorch for the paper.

- It only supports Pytorch backend now.
# Dataset Statics

| Dataset   | # Nodes | # Edges | # Classes |
|-----------|---------|---------|-----------|
| Cora      | 2,708   | 10,556  | 7         |
| Citeseer  | 3,327   | 9,228   | 6         |
| Pubmed    | 19,717  | 88,651  | 3         |
| CS        | 2,708   | 10,556  | 7         |
| Physics   | 3,327   | 9,228   | 6         |
| Computers | 13,752  | 491,722 | 10        |
| Photo     | 7,650   | 238,162 | 8         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid), [Coauthor](https://gammagl.readthedocs.io/en/latest/generated/gammagl.datasets.Coauthor.html) and [Amazon](https://gammagl.readthedocs.io/en/latest/generated/gammagl.datasets.Amazon.html).


Results
-------

```bash
# available dataset: "Cora", "Citeseer", "Pubmed", "CS", "Physics", "Computers", "Photo"
TL_BACKEND="torch" python DR-GST.py --dataset Cora --model GCN --labelrate 20 --drop_method dropout --droprate 0.3
TL_BACKEND="torch" python DR-GST.py --dataset Citeseer --model GCN --labelrate 20 --drop_method dropout --droprate 0.3
TL_BACKEND="torch" python DR-GST.py --dataset Pubmed --model GCN --labelrate 20 --drop_method dropout --droprate 0.3
TL_BACKEND="torch" python DR-GST.py --dataset CS --model GCN --labelrate 20 --drop_method dropout --droprate 0.3
TL_BACKEND="torch" python DR-GST.py --dataset Physics --model GCN --labelrate 20 --drop_method dropout --droprate 0.3
TL_BACKEND="torch" python DR-GST.py --dataset Computers --model GCN --labelrate 20 --drop_method dropout --droprate 0.3
TL_BACKEND="torch" python DR-GST.py --dataset Photo --model GCN --labelrate 20 --drop_method dropout --droprate 0.3
```


| Dataset  | Paper Code | Out(th) |
|----------|------------|---------|
| Cora     | 83.34      | 92.00   |
| CiteSeer | 75.78      | 92.80   |
| PubMed   | 81.08      | 91.20   | 
| CoraFull | 62.75      | 91.00   |
| Flickr   | 53.66      | 92.80   |
