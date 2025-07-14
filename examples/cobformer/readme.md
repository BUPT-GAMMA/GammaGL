# Less is More: on the Over-Globalizing Problem in Graph Transformers (CoBFormer)

- Paper link: [http://www.shichuan.org/doc/177.pdf](http://www.shichuan.org/doc/177.pdf)
- Author's code repo: [https://github.com/BUPT-GAMMA/CoBFormer](https://github.com/BUPT-GAMMA/CoBFormer)

# Dataset Statics

| Dataset       | # Nodes   | # Edges    | # Feats | Edge hom | # Classes |
|---------------|-----------|------------|---------|----------|-----------|
| Cora          | 2,708     | 5,429      | 1,433   | 0.83     | 7         |
| CiteSeer      | 3,327     | 4,732      | 3,703   | 0.72     | 6         |
| PubMed        | 19,717    | 44,338     | 500     | 0.79     | 3         |
| Actor         | 7,600     | 26,752     | 931     | 0.22     | 5         |
| Deezer        | 28,281    | 92,752     | 31,241  | 0.52     | 2         |
| Ogbn-Arxiv    | 169,343   | 1,166,343  | 128     | 0.63     | 40        |
| Ogbn-Products | 2,449,029 | 61,859,140 | 100     | 0.81     | 47        |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

# Results

- Available dataset: "Cora", "Citeseer", "Pubmed", "film", "deezer", "ogbn-arxiv", "ogbn-products"

```bash
# available dataset: "cora", "citeseer", "pubmed"
python cobformer_trainer.py --dataset=Cora --learning_rate=0.01 --gcn_wd=1e-3 --weight_decay=5e-5 --gcn_type=1 --gcn_layers=2 --n_patch=112 --use_patch_attn --alpha=0.7 --tau=0.3 --gpu_id=3
python cobformer_trainer.py --dataset=CiteSeer --learning_rate=5e-3 --gcn_wd=1e-2 --weight_decay=5e-5 --gcn_type=1 --gcn_layers=2 --n_patch=144 --use_patch_attn --alpha=0.8 --tau=0.7 --gpu_id=3
python cobformer_trainer.py --dataset=PubMed --learning_rate=5e-3 --gcn_wd=1e-3 --weight_decay=1e-3 --gcn_type=1 --gcn_layers=2 --n_patch=224 --use_patch_attn --alpha=0.7 --tau=0.3 --gpu_id=3
python cobformer_trainer.py --dataset=film --learning_rate=5e-2 --gcn_wd=1e-4 --weight_decay=1e-3 --gcn_type=1 --gcn_layers=2 --n_patch=112 --use_patch_attn --alpha=0.7 --tau=0.9 --gpu_id=3
python cobformer_trainer.py --dataset=deezer --learning_rate=0.01 --gcn_wd=1e-3 --weight_decay=5e-4 --gcn_type=1 --gcn_layers=2 --n_patch=224 --use_patch_attn --alpha=0.8 --tau=0.9 --gpu_id=3
python cobformer_trainer.py --dataset=ogbn-arxiv --learning_rate=1e-3 --weight_decay=0. --gcn_use_bn --gcn_type=2 --gcn_layers=3 --n_patch=2048 --use_patch_attn --alpha=0.9 --tau=0.9 --gpu_id=3
python cobformer_trainer.py --dataset=ogbn-products --learning_rate=5e-4 --weight_decay=0. --gcn_type=2 --gcn_layers=3 --gcn_use_bn --n_patch=8192 --use_patch_attn --batch_size=150000 --alpha=0.9 --tau=0.7 --gpu_id=3
```

- Or use `runs.sh` to run all experiments

```bash
bash runs.sh
```

Paper:

| Dataset           | CoB-G Mi-F1    | CoB-T Mi-F1    | CoB-G Ma-F1    | CoB-T Ma-F1    |
|-------------------|----------------|----------------|----------------|----------------|
| **Cora**          | 84.96 ± 0.34 % | 85.28 ± 0.16 % | 83.52 ± 0.15 % | 84.10 ± 0.28 % |
| **CiteSeer**      | 74.68 ± 0.33 % | 74.52 ± 0.48 % | 69.73 ± 0.45 % | 69.82 ± 0.55 % |
| **PubMed**        | 80.52 ± 0.25 % | 81.42 ± 0.53 % | 80.02 ± 0.28 % | 81.04 ± 0.49 % |
| **Actor**         | 31.05 ± 1.02 % | 37.41 ± 0.36 % | 27.01 ± 1.77 % | 34.96 ± 0.68 % |
| **Deezer**        | 63.76 ± 0.62 % | 66.96 ± 0.37 % | 62.32 ± 0.94 % | 65.63 ± 0.36 % |
| **Ogbn-Arxiv**    | 73.17 ± 0.18 % | 72.76 ± 0.11 % | 52.31 ± 0.40 % | 51.64 ± 0.09 % |
| **Ogbn-Products** | 78.09 ± 0.16 % | 78.15 ± 0.07 % | 38.21 ± 0.22 % | 37.91 ± 0.44 % |

Our:

| Dataset           | CoB-G Mi-F1    | CoB-T Mi-F1    | CoB-G Ma-F1    | CoB-T Ma-F1    |
|-------------------|----------------|----------------|----------------|----------------|
| **Cora**          | 84.44 ± 0.60 % | 84.42 ± 0.79 % | 83.23 ± 0.93 % | 83.41 ± 0.45 % |
| **CiteSeer**      | 74.64 ± 0.40 % | 74.40 ± 0.52 % | 69.76 ± 0.37 % | 69.93 ± 0.59 % |
| **PubMed**        | 80.30 ± 0.32 % | 81.02 ± 0.47 % | 79.72 ± 0.35 % | 80.66 ± 0.52 % |
| **Actor**         | 30.44 ± 1.16 % | 34.19 ± 4.05 % | 24.42 ± 3.25 % | 28.08 ± 8.36 % |
| **Deezer**        | 64.05 ± 0.42 % | 66.62 ± 0.81 % | 63.12 ± 0.46 % | 65.03 ± 0.39 % |
| **Ogbn-Arxiv**    | 73.09 ± 0.10 % | 72.71 ± 0.16 % | 52.37 ± 0.33 % | 51.39 ± 0.21 % |
| **Ogbn-Products** | 78.14 ± 0.09 % | 78.15 ± 0.08 % | 38.36 ± 0.19 % | 38.09 ± 0.35 % |

