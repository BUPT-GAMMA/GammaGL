# Less is More: on the Over-Globalizing Problem in Graph Transformers (CoBformer)

- Paper link: [Less is More: on the Over-Globalizing Problem in Graph Transformers](http://arxiv.org/abs/2405.01102)

## Dataset Statistics

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

## Results
```bash
# Cora
python cobformer_trainer.py --dataset=Cora --learning_rate=0.01 --gcn_wd=1e-3 --weight_decay=5e-5 --gcn_type=1 --gcn_layers=2 --n_patch=112 --use_patch_attn --alpha=0.7 --tau=0.3 --gpu_id=0

# CiteSeer
python cobformer_trainer.py --dataset=CiteSeer --learning_rate=5e-3 --gcn_wd=1e-2 --weight_decay=5e-5 --gcn_type=1 --gcn_layers=2 --n_patch=144 --use_patch_attn --alpha=0.8 --tau=0.7 --gpu_id=0

# PubMed
python cobformer_trainer.py --dataset=PubMed --learning_rate=5e-3 --gcn_wd=1e-3 --weight_decay=1e-3 --gcn_type=1 --gcn_layers=2 --n_patch=224 --use_patch_attn --alpha=0.7 --tau=0.3 --gpu_id=0

```

| Dataset  | Paper | Our(tf)     | 
| -------- | ----- | ----------- | 
| cora     | 85.28 | 83.16 ± 0.59 |
| citeseer | 74.52 | 71.20 ± 0.80 | 
| pubmed   | 81.42 | 81.84 ± 0.45 |

