# Distribution Recovered Graph Self-Training framework (DR-GST)

- Paper link: [https://arxiv.org/abs/2201.11349](https://arxiv.org/abs/2201.11349)
- Author's code repo: [https://github.com/TaurusTaurus-Rui/DR-GST](https://github.com/TaurusTaurus-Rui/DR-GST). Note that the original code is implemented with Pytorch for the paper.

# Dataset Statics

| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |
| Flickr   | ---     | 899576  | 7         |


 
Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid), [Flickr](https://gammagl.readthedocs.io/en/latest/generated/gammagl.datasets.Flickr.html)


Results
-------

```bash
# available dataset: "Cora", "Citeseer", "Pubmed", "CS", "Physics", "Computers", "Photo"
TL_BACKEND="torch" python DR-GST.py --dataset Cora --model GCN --labelrate 20 --drop_method dropout --droprate 0.3
TL_BACKEND="torch" python DR-GST.py --dataset CiteSeer --model GCN --labelrate 20 --drop_method dropout --droprate 0.3
TL_BACKEND="torch" python DR-GST.py --dataset PubMed --model GCN --labelrate 20 --drop_method dropout --droprate 0.3
TL_BACKEND="torch" python DR-GST.py --dataset CoraFull --model GCN --labelrate 20 --drop_method dropout --droprate 0.3
TL_BACKEND="torch" python DR_GST.py --model GCN --labelrate 5 --drop_method dropout --droprate 0.4 --dataset Flickr--threshold 0.8 --weight_decay 5e-5 --lr 0.005
‘’‘


| Dataset  | Paper Code | Out(th) |
|----------|------------|---------|
| Cora     | 83.34      | 88.60   |
| CiteSeer | 75.78      | 77.60   |
| PubMed   | 81.08      | 88.70   | 
| CoraFull | 62.75      | 80.30   |
| Flickr   | 37.84      | 37.59   |
