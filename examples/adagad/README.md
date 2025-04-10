# Graph Convolutional Networks (GCN)

  

- Paper link: [https://arxiv.org/abs/2312.14535](https://arxiv.org/abs/2312.14535)

- Author's code repo: [jweihe/ADA-GAD: Official PyTorch implementation for the paper ADA-GAD: Anomaly-Denoised Autoencoders for Graph Anomaly Detection (AAAI 2024).](https://github.com/jweihe/ADA-GAD). 
  Note that the original code is implemented with Pytorch for the paper.

  

# Dataset Statics

Dataset links:
- Original version: [pygod-team/data: Data repository for PyGOD](https://github.com/pygod-team/data)
- .npz version（used in this examples）: [SharkRemW/data: Data repository for PyGOD，add '.npz' version](https://github.com/SharkRemW/data)

| Dataset      | Type     | #Nodes | #Edges  | \#Feat | Avg. Degree | #Con. | #Strct. | #Outliers | Outlier Ratio |
| ------------ | -------- | ------ | ------- | ------ | ----------- | ----- | ------- | --------- | ------------- |
| 'reddit'     | organic  | 10,984 | 168,016 | 64     | 15.3        | -     | -       | 366       | 3.3%          |
| 'inj_cora'   | injected | 2,708  | 11,060  | 1,433  | 4.1         | 70    | 70      | 138       | 5.1%          |
| 'inj_amazon' | injected | 13,752 | 515,042 | 767    | 37.2        | 350   | 350     | 694       | 5.0%          |

Refer to [ADDataset](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

  

Results

-------

  

```bash

# available dataset: "inj_cora", "inj_amazon", "reddit"
TL_BACKEND="torch" python adagad_trainer.py --use_cfg --seeds 0 --dataset inj_cora

TL_BACKEND="torch" python adagad_trainer.py --use_cfg --seeds 0 --dataset inj_amazon

TL_BACKEND="torch" python adagad_trainer.py --use_cfg --seeds 0 --dataset reddit
```

Criteria：AUC

| Dataset    | Paper      | Our(th)    |
| ---------- | ---------- | ---------- |
| inj_cora   | 84.73±0.01 | 85.67±0.22 |
| inj_amazon | 83.25±0.03 | 81.01±0.13 |
| reddit     | 56.89±0.01 | 56.86±0.12 |

![[auc_comparison_paper_vs_our.png]]