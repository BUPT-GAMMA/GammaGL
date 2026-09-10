# GNNPP

- Paper link: [https://arxiv.org/abs/2502.09263](https://arxiv.org/abs/2502.09263)
- Author's code repo: [https://github.com/LUOyk1999/GNNPlus](https://github.com/LUOyk1999/GNNPlus)

## Dataset Statics

| Dataset | # Train Graphs | # Validation Graphs | Task |
|---|---:|---:|---|
| ZINC-subset | 10,000 | 1,000 | Regression |
| Refer to [ZINC](https://gammagl.readthedocs.io/en/stable/generated/gammagl.datasets.ZINC.html). | | | |
 
## Results

```bash
TL_BACKEND="pytorch" python gnnpp_trainer.py \
    --dataset_path ./data/ZINC \
    --batch_size 32 \
    --emb_dim 70 \
    --num_layers 9 \
    --dropout 0.05 \
    --lr 0.0002 \
    --n_epoch 2000
```

| Dataset | Paper | Ours |
|---|---:|---:|
| ZINC-subset | 0.077 | 0.079 ± 0.004 |

> **Note:** Our implementation is slightly different from the author's implementation in the optimizer. The original implementation applies different weight decay coefficients to different learnable parameters, while TensorLayerX currently does not support parameter-wise weight decay.