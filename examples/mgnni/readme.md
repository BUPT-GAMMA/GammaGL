# MGNNI: Multiscale Graph Neural Networks with Implicit Layers (NGNNI)

- Paper link: [https://arxiv.org/abs/2210.08353](https://arxiv.org/abs/2210.08353)
- Author's code repo: [https://github.com/liu-jc/MGNNI](https://github.com/liu-jc/MGNNI). Note that the original code is 
  implemented with Pytorch for the paper. 

# Dataset Statics

| Dataset   | # Nodes | # Edges | # Classes |
|-----------|---------|---------|-----------|
| Cornell   | 183     | 280     | 5         |
| Texas     | 183     | 295     | 5         |
| Wisconsin | 251     | 466     | 5         |

Refer to [WebKB](https://gammagl.readthedocs.io/en/latest/generated/gammagl.datasets.WebKB.html#gammagl.datasets.WebKB).

Results
-------

```bash
# available dataset: "cora", "citeseer", "pubmed"
TL_BACKEND="paddle" python mgnni_trainner.py --dataset cornell --lr 0.5 --l2_coef 5e-6 --model MGNNI_m_att --ks [1,2] --epochs 300
TL_BACKEND="paddle" python mgnni_trainner.py --dataset texas --lr 0.5 --l2_coef 5e-6 --model MGNNI_m_att --ks [1,2] --epochs 300
TL_BACKEND="paddle" python mgnni_trainner.py --dataset wisconsin --lr 0.5 --l2_coef 5e-6 --model MGNNI_m_att --ks [1,2] --epochs 300
TL_BACKEND="tensorflow" python mgnni_trainner.py --dataset cornell --lr 0.5 --l2_coef 5e-6 --model MGNNI_m_att --ks [1,2] --epochs 300
TL_BACKEND="tensorflow" python mgnni_trainner.py --dataset texas --lr 0.5 --l2_coef 5e-6 --model MGNNI_m_att --ks [1,2] --epochs 300
TL_BACKEND="tensorflow" python mgnni_trainner.py --dataset wisconsin --lr 0.5 --l2_coef 5e-6 --model MGNNI_m_att --ks [1,2] --epochs 300
TL_BACKEND="torch" python mgnni_trainner.py --dataset cornell --lr 0.5 --l2_coef 5e-6 --model MGNNI_m_att --ks [1,2] --epochs 300
TL_BACKEND="torch" python mgnni_trainner.py --dataset texas --lr 0.5 --l2_coef 5e-6 --model MGNNI_m_att --ks [1,2] --epochs 300
TL_BACKEND="torch" python mgnni_trainner.py --dataset wisconsin --lr 0.5 --l2_coef 5e-6 --model MGNNI_m_att --ks [1,2] --epochs 300 
```

| Dataset   | Paper | Our(pd)    | Our(tf)    | Our(th)    | Our(ms) |
|-----------|-------|------------|------------|------------|---------|
| Cornell   | 78.38 | 78.92±1.21 | 78.38±0.00 | 78.38±0.00 |         |
| Texas     | 81.08 | 85.95±1.21 | 83.78±0.00 | 84.86±1.48 |         |
| Wisconsin | 82.35 | 83.53±1.07 | 82.35±0.00 | 83.92±0.88 |         |
