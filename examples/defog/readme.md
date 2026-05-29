# DeFoG: Discrete Flow Matching for Graph Generation

- Paper: [https://arxiv.org/abs/2410.04263](https://arxiv.org/abs/2410.04263)
- Original code: [https://github.com/manuelmlmadeira/DeFoG](https://github.com/manuelmlmadeira/DeFoG)

This directory contains the GammaGL reproduction of DeFoG. The current implementation uses a shared training / validation-sampling / final-sampling pipeline, with validation-driven best-checkpoint selection and optional EMA swapping during evaluation and sampling.

## Supported Datasets

| Dataset | Type | Node Types | Edge Types | Conditional |
|---------|------|------------|------------|-------------|
| synthetic | Synthetic | configurable | configurable | -- |
| planar | Synthetic | 2 | 2 | -- |
| tree | Synthetic | 2 | 2 | -- |
| sbm | Synthetic | 2 | 2 | -- |
| qm9 | Molecular | 4 | 5 | mu / homo / both |
| guacamol | Molecular | 12 | 5 | -- |
| zinc250k | Molecular | 9 | 4 | -- |
| moses | Molecular | 8 | 5 | -- |
| tls | Molecular | 9 | 2 | half-half |

## File Structure

| File | Description |
| ---- | ----------- |
| `defog_trainer.py` | Main entry for training, validation sampling, final sampling, and evaluation |
| `defog_sample_only.py` | Sampling-only entry that loads `best` / `last` checkpoints |
| `defog_utils.py` | Dense conversion, placeholder utilities, EMA, backend helpers |
| `flow_utils.py` | Flow interpolation terms |
| `flow_matching_utils.py` | Discrete feature sampling and noise generation |
| `noise_distribution.py` | Noise distributions / transitions |
| `rate_matrix.py` | CTMC rate matrix construction |
| `time_distorter.py` | Training / sampling time distortion |
| `extra_features.py` | Structural features (RRWP, cycles, eigen features) |
| `extra_features_molecular.py` | Molecular domain features |
| `train_metrics.py` | Cross-entropy / KLD training metrics |
| `rdkit_functions.py` | Molecular metrics and SMILES helpers |
| `spectre_utils.py` | Synthetic graph evaluation |
| `visualization.py` | Graph and molecule visualization |

## Model Components

| File | Description |
| ---- | ----------- |
| `gammagl/models/defog.py` | `DeFoGModel` Graph Transformer denoiser |
| `gammagl/layers/attention/defog_layer.py` | `XEyTransformerLayer`, `NodeEdgeBlock`, `Xtoy`, `Etoy` |

## Preset Behavior

For named datasets (`planar`, `tree`, `sbm`, `qm9`, `guacamol`, `zinc250k`, `moses`), `defog_trainer.py` and `defog_sample_only.py` automatically apply DeFoG-aligned dataset presets through `apply_dataset_preset()`.

- Presets set dataset-specific values such as `n_layers`, `batch_size`, `sample_steps`, validation cadence, and sampling distortion.
- Explicit CLI flags override the preset values.
- `tls` and `synthetic` do not receive these preset overrides.

## Checkpoint Semantics

Training writes paired model / EMA snapshots:

- `last_model.npz` / `last_ema.pkl`
- `best_model.npz` / `best_ema.pkl`

Best-checkpoint selection is validation-driven:

1. save `last`
2. run validation sampling
3. evaluate generated graphs
4. compute `selection_score`
5. update `best` if the validation score improves

Sampling prefers `best_model.npz`; if it does not exist, it falls back to `last_model.npz`.

## Quick Start

Run commands from `examples/defog`.

```bash
# Quick smoke test on synthetic data
TL_BACKEND="torch" python defog_trainer.py \
  --n_layers 2 \
  --n_epochs 3 \
  --batch_size 4 \
  --sample \
  --sample_steps 5 \
  --num_samples 3 \
  --num_graphs 20

# Planar training with preset hyperparameters (single run)
TL_BACKEND="torch" python defog_trainer.py --dataset planar --data_root ./datasets --sample --evaluate

# Planar 3-seed reproduction
for seed in 43 44 45; do
  TL_BACKEND="torch" python defog_trainer.py \
    --dataset planar \
    --data_root ./datasets \
    --seed $seed \
    --save_dir ./checkpoints_planar_seed${seed} \
    --sample \
    --evaluate
done

# Tree / SBM training with preset hyperparameters
TL_BACKEND="torch" python defog_trainer.py --dataset tree --data_root ./datasets --sample --evaluate
TL_BACKEND="torch" python defog_trainer.py --dataset sbm --data_root ./datasets --sample --evaluate

# QM9 training with preset hyperparameters
TL_BACKEND="torch" python defog_trainer.py --dataset qm9 --data_root ./datasets --sample --evaluate

# QM9 conditional generation
TL_BACKEND="torch" python defog_trainer.py \
  --dataset qm9 \
  --data_root ./datasets \
  --conditional \
  --target mu \
  --guidance_weight 2.0 \
  --sample \
  --evaluate

# Sampling only from an existing checkpoint directory
TL_BACKEND="torch" python defog_sample_only.py \
  --dataset planar \
  --data_root ./datasets \
  --save_dir ./checkpoints_planar \
  --evaluate

# EMA + multi-fold sampling evaluation
TL_BACKEND="torch" python defog_sample_only.py \
  --dataset qm9 \
  --data_root ./datasets \
  --save_dir ./checkpoints_qm9 \
  --ema_decay 0.999 \
  --num_sample_fold 3 \
  --evaluate
```

## Benchmark Results

<!--
### Planar (3 seeds)

We train DeFoG on the Planar dataset for 100,000 epochs with 3 different random seeds (43, 44, 45) and report mean ± std. Results are compared against the DeFoG paper (Table 7, mean ± std over 5 sampling runs with 40 graphs each).

| Metric | Paper (DeFoG) | Seed 43 | Seed 44 | Seed 45 | Mean ± Std |
|--------|---------------|---------|---------|---------|------------|
| Valid ↑ | 99.5 ± 1.0 | 99.0 | 100.0 | 99.0 | **99.3 ± 0.5** |
| Unique ↑ | 100.0 ± 0.0 | 100.0 | 100.0 | 100.0 | **100.0 ± 0.0** |
| Novel ↑ | 100.0 ± 0.0 | 100.0 | 100.0 | 100.0 | **100.0 ± 0.0** |
| V.U.N. ↑ | 99.5 ± 1.0 | 99.0 | 100.0 | 99.0 | **99.3 ± 0.5** |
| Planar Acc ↑ | — | 0.9900 | 1.0000 | 0.9900 | **0.9933 ± 0.0047** |
| Degree ↓ | 0.0005 ± 0.0002 | 0.000718 | 0.000700 | 0.000523 | **0.0006 ± 0.0001** |
| Clustering ↓ | 0.0501 ± 0.0149 | 0.044833 | 0.050792 | 0.058708 | **0.0514 ± 0.0057** |
| Orbit ↓ | 0.0006 ± 0.0004 | 0.000811 | 0.002144 | 0.000882 | **0.0013 ± 0.0006** |
| Spectre ↓ | 0.0072 ± 0.0011 | 0.008865 | 0.006475 | 0.007335 | **0.0076 ± 0.0010** |
| Wavelet ↓ | 0.0014 ± 0.0002 | 0.000327 | 0.000325 | 0.000094 | **0.0002 ± 0.0001** |
| Ratio ↓ | 1.6 ± 0.4 | 1.0000 | 1.0000 | 1.0000 | **1.0000 ± 0.0000** |

*Training command (per seed):*
```bash
TL_BACKEND="torch" python defog_trainer.py \
  --dataset planar \
  --data_root ./datasets \
  --seed 43 \
  --save_dir ./checkpoints_planar_seed43 \
  --sample \
  --evaluate
```
-->

### Planar (seed 0, completed)

Trained for 100,000 epochs. Best checkpoint at epoch 96000 (validation score 1.0).

| Metric | Best (epoch 96000) |
|--------|--------------------|
| Valid ↑ | 100.0% |
| Unique ↑ | 100.0% |
| Non-iso ↑ | 100.0% |
| Planar Acc ↑ | 100.0% |
| Degree ↓ | 0.000566 |
| Spectre ↓ | 0.009673 |
| Clustering ↓ | 0.029575 |
| Orbit ↓ | 0.000386 |
| Wavelet ↓ | 0.000057 |

### Planar (seed 1, ongoing)

Resume from epoch 3935, currently ~54700 / 100000.

| Metric | Current (epoch 54000) |
|--------|----------------------|
| Valid ↑ | 92.5% |
| Unique ↑ | 100.0% |
| Non-iso ↑ | 100.0% |
| Planar Acc ↑ | 92.5% |
| Degree ↓ | 0.000504 |
| Spectre ↓ | 0.009354 |
| Clustering ↓ | 0.030809 |
| Orbit ↓ | 0.004450 |
| Wavelet ↓ | 0.000093 |

### Planar (seed 2, planned)

Planned for future reproduction.

### Tree (seed 0, completed)

Trained for 100,000 epochs. Best checkpoint at epoch 72000 (validation score 1.0).

| Metric | Best (epoch 72000) | Final (epoch 100000) |
|--------|--------------------|----------------------|
| Valid ↑ | **100.0%** | 55.0% |
| Unique ↑ | 82.5% | 97.5% |
| Non-iso ↑ | 100.0% | 100.0% |
| Tree Acc ↑ | **100.0%** | 55.0% |
| Degree ↓ | 0.000575 | 0.000314 |
| Spectre ↓ | 0.010322 | 0.010755 |
| Clustering ↓ | 0.000000 | 0.000000 |
| Orbit ↓ | 0.000007 | 0.000017 |
| Wavelet ↓ | 0.000613 | 0.000479 |

### QM9 no-H (seed 0, completed)

Trained for 1,000 epochs without explicit hydrogens.

| Metric | GammaGL (no-H) | Paper (DeFoG, no-H 500 steps) |
|--------|----------------|-------------------------------|
| Validity ↑ | **99.02%** | 99.3% |
| Relaxed Validity ↑ | **99.80%** | 99.4% |
| Uniqueness ↑ | **99.80%** | 96.3% |
| Novelty ↑ | **40.39%** | — |
| FCD ↓ | **0.7934** | 0.12 |

*Training command:*
```bash
TL_BACKEND="torch" python defog_trainer.py \
  --dataset qm9 \
  --data_root ./datasets \
  --save_dir ./checkpoints_qm9_noh_seed0_final \
  --seed 0 \
  --gpu 5 \
  --n_layers 9 \
  --n_epochs 1000 \
  --batch_size 1024 \
  --lr 2e-4 \
  --train_distortion identity \
  --sample_distortion polydec \
  --sample_steps 500 \
  --omega 0 \
  --eta 0 \
  --check_val_every_n_epochs 50 \
  --sample_every_val 1 \
  --val_num_samples 512 \
  --remove_h
```

## Important Parameters

The parser-level defaults are generic. For named datasets, presets may replace them unless you pass an explicit CLI override.

| Parameter | Parser Default | Description |
| --------- | -------------- | ----------- |
| `--dataset` | `synthetic` | Dataset name |
| `--data_root` | `None` | Root directory for real datasets |
| `--conditional` | off | Enable classifier-free guidance (QM9 only) |
| `--target` | `mu` | Conditional target: `mu` / `homo` / `both` |
| `--guidance_weight` | `2.0` | CFG weight |
| `--n_layers` | `5` | Transformer depth |
| `--batch_size` | `32` | Training batch size |
| `--n_epochs` | `100` | Training epochs |
| `--lr` | `2e-4` | Learning rate |
| `--weight_decay` | `1e-12` | AdamW weight decay |
| `--ema_decay` | `0.0` | EMA decay (`0` disables EMA) |
| `--grad_clip_norm` | `1.0` | Gradient clipping norm |
| `--kld` | off | Use KLD for node / edge losses |
| `--lambda_E` | `5.0` | Edge loss weight |
| `--lambda_y` | `0.0` | Global-property loss weight |
| `--transition` | `marginal` | Noise transition |
| `--extra_features` | `rrwp` | Extra structural features |
| `--rrwp_steps` | `12` | RRWP steps |
| `--train_distortion` | `identity` | Training time distortion |
| `--sample` | off | Run final sampling after training |
| `--evaluate` | off | Evaluate generated graphs |
| `--sample_steps` | `100` | Number of denoising steps |
| `--sample_distortion` | `identity` | Sampling time distortion |
| `--num_samples` | `20` | Number of generated graphs |
| `--num_sample_fold` | `1` | Number of sampling folds |
| `--sample_every_val` | `0` | Run validation sampling every N validation events |
| `--check_val_every_n_epochs` | `0` | Run validation cadence every N epochs |
| `--val_num_samples` | `40` | Number of samples used in validation selection |
| `--eta` | `0.0` | R^db strength |
| `--omega` | `0.0` | R^tg strength |
| `--rdb` | `general` | RDB design |
| `--rdb_crit` | `max_marginal` | RDB sub-criterion |
| `--save_dir` | `./checkpoints` | Output checkpoint directory |

## Evaluation Outputs

### Synthetic (`planar`, `tree`, `sbm`)
Typical outputs include:

- `planar_acc` / `tree_acc`
- `frac_unique`
- `frac_non_iso`
- `frac_unique_non_iso`
- `frac_unic_non_iso_valid`
- compatibility aliases under `sampling/...`
- `selection_score`

### Molecular (`qm9`, `guacamol`, `zinc250k`, `moses`, `tls`)
Typical outputs include:

- `Validity`
- `Relaxed Validity`
- `Uniqueness`
- `Novelty`
- `fcd`
- distribution MAE terms
- `selection_score`

## Notes

- `defog_sample_only.py` must use model hyperparameters compatible with the saved checkpoint. If you rely on dataset presets, keep the dataset name consistent with the training run.
- For reproducibility checks, prefer evaluating checkpoints produced by the current training code rather than mixing in older checkpoints created before the validation / checkpoint / metric-key fixes.
