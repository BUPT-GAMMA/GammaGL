# CoED-GNN Node Classification

- Paper link: [https://arxiv.org/abs/2410.14109](https://arxiv.org/abs/2410.14109)
- Author's code repo: [https://github.com/hormoz-lab/coed-gnn](https://github.com/hormoz-lab/coed-gnn)

## Dataset Statics

| Dataset    | # Nodes | # Edges | # Classes |
|------------|---------|---------|-----------|
| Cora       | 2,708   | 10,556  | 7         |
| Texas      | 183     | 309     | 5         |
| Wisconsin  | 251     | 515     | 5         |
| Chameleon  | 2,277   | 36,101  | 5         |
| Squirrel   | 5,201   | 217,073 | 5         |

All datasets use the `Geom-GCN` 10 fixed splits for evaluation.

## Files

- `examples/coed/coed_trainer.py`: Multi-dataset training and evaluation entry
- `gammagl/models/coed.py`: CoED-GNN backbone model
- `gammagl/layers/conv/coed_conv.py`: CoED directional convolution layer

## Results

### Cora

```bash
TL_BACKEND="torch" python examples/coed/coed_trainer.py --dataset cora
```

| Metric     | Paper      | Our(torch)           |
|------------|------------|----------------------|
| Test Acc   | 86.42      | 87.00 +/- 1.44      |

### Texas

```bash
TL_BACKEND="torch" python examples/coed/coed_trainer.py --dataset texas
```

| Metric     | Paper      | Our(torch)           |
|------------|------------|----------------------|
| Test Acc   |            |                      |

### Wisconsin

```bash
TL_BACKEND="torch" python examples/coed/coed_trainer.py --dataset wisconsin
```

| Metric     | Paper      | Our(torch)           |
|------------|------------|----------------------|
| Test Acc   |            |                      |

### Chameleon

```bash
TL_BACKEND="torch" python examples/coed/coed_trainer.py --dataset chameleon
```

| Metric     | Paper      | Our(torch)           |
|------------|------------|----------------------|
| Test Acc   |            |                      |

### Squirrel

```bash
TL_BACKEND="torch" python examples/coed/coed_trainer.py --dataset squirrel
```

| Metric     | Paper      | Our(torch)           |
|------------|------------|----------------------|
| Test Acc   |            |                      |

## Notes

- The default setup uses `hidden_dim=64`, `num_layers=2`, `lr=1e-3`, `l2_coef=0.0`, `alpha=0.5`, `self_loop=True`, `normalize=False`, `self_feature_transform=False`, `patience=100`, `n_epoch=5000`.
- The implementation evaluates all 10 Geom-GCN fixed splits and reports mean +/- std test accuracy.
- The model and convolution layers are registered in `gammagl/models/__init__.py` and `gammagl/layers/conv/__init__.py` and can be imported via standard GammaGL paths.
