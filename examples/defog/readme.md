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
| `defog_utils.py` | Dense conversion, placeholder utilities, EMA, backend helpers |
| `flow_matching.py` | Mathematical engine for CTMC rate matrix, discrete flow matching, noise distribution, and time distorter |

| `dataset_utils.py` | Dataset loading and dataset-info computation |
| `evaluator.py` | Evaluation driver (validity, uniqueness, novelty, FCD, SPECTRE) |
| `sampler.py` | CTMC forward / backward sampling |
| `extra_features.py` | Structural and molecular features (RRWP, cycles, eigen features, etc.) |
| `train_metrics.py` | Cross-entropy / KLD training metrics |
| `rdkit_functions.py` | Molecular metrics and SMILES helpers |
| `spectre_utils.py` | Synthetic graph evaluation |
| `multi_gpu.py` | Multi-GPU training helpers |

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

## Dependencies & Architecture Boundary

**GammaGL Core Integration:**
- The graph neural network structure (`gammagl/models/defog.py`, `gammagl/layers/attention/defog_layer.py`) and dataset classes (`gammagl/datasets/...`) are strictly isolated from heavy domain-specific dependencies.
- You do **not** need `rdkit`, `graph-tool`, or `networkx` to import GammaGL core modules. They are gracefully caught or lazily imported.

**Examples Sandbox (`examples/defog`):**
- **Why is there so much custom code?** DeFoG is the *first* Discrete Flow Matching model in GammaGL. It requires a Continuous-Time Markov Chain (CTMC) flow matching solver, discrete loss computation, and complex evaluation protocols that are not standard classification/regression tasks. To avoid bloating the GammaGL core with flow-matching specifics, these components are kept in `examples/defog`.
- **Optional Dependencies (see `requirements.txt`)**:
  - `rdkit` and `fcd`: Required **only** for molecular dataset evaluation. If missing, it will gracefully warn and skip molecular metrics.
  - `graph-tool` and `scipy`: Required **only** for SBM graph validation. If missing, SBM evaluation will raise an `ImportError` instructing the user to install them (preferably via `conda install -c conda-forge graph-tool scipy`).
  - `pyemd`, `networkx`, `scipy`: Used for SPECTRE evaluation metrics.
- Running complete generation evaluation on molecular datasets requires `rdkit` and potentially `orca` / `graph-tool` as specified by the original authors.

**Supported Backend:**
- This implementation currently supports **`TL_BACKEND=torch` only**. Other TensorLayerX backends (TensorFlow, PaddlePaddle, MindSpore) have not been tested and are not guaranteed to work. The model operations rely on specific PyTorch sparse and broadcasting behaviors.
- Note: `multi_gpu.py` is an experimental, pure PyTorch multi-GPU wrapper intended only for users with heavy Torch environments.

## Quick Start / Minimal Smoke Test

We provide minimal, CPU-friendly smoke test commands (1 epoch, minimal synthetic data). This verifies that the entire flow-matching training loop runs properly on the currently supported **TensorLayerX `torch` backend** (`TL_BACKEND=torch`) without needing GPU or heavy dependencies like `rdkit`/`graph-tool`.

```bash
cd examples/defog

# 方案A：两步（训练 + 采样）
# 先训练 1 epoch
TL_BACKEND="torch" python defog_trainer.py \
  --dataset synthetic \
  --n_layers 1 \
  --n_epochs 1 \
  --batch_size 2 \
  --num_graphs 10 \
  --gpu -1

# 再基于刚保存的 checkpoint 做最小采样验证
TL_BACKEND="torch" python defog_sample_only.py \
  --dataset synthetic \
  --n_layers 1 \
  --sample_steps 2 \
  --num_samples 2 \
  --gpu -1 \
  --model_path checkpoints/last_model.npz
```

## Advanced Examples

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

## Dependency Management
To keep GammaGL lightweight, DeFoG evaluates synthetic graphs and molecules using specialized external libraries which are **not** installed by default.

### Optional Evaluation Dependencies
If you want to perform full evaluation on `spectre` or molecular datasets (`qm9`, `zinc250k`):
```bash
# For synthetic graph evaluation (SPECTRE)
pip install pyemd scipy networkx

# For molecular evaluation (QM9, ZINC250k)
pip install rdkit fcd
```
If these dependencies are missing, the training will still run normally but the evaluation metrics will be skipped and output `-1` or `NaN`.

## Minimal CPU Smoke Test
You can verify the model is functioning correctly without any heavy dependencies by running a minimal smoke test on a small synthetic dataset:
```bash
python defog_trainer.py --dataset synthetic --n_epochs 1 --batch_size 2 --sample_steps 2 --num_graphs 4 --n_layers 2
```
Expected output will show the dataset building dynamically and the loss being printed, followed by completion without crashing. Alternatively, you can run the provided smoke test script:
```bash
python tests/models/test_defog_smoke.py
```
This ensures that the GammaGL core layers and flow matching engine are backend-neutral and do not suffer from any hard `torch` or `rdkit` import issues.

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

### Planar (3 seeds, completed)

Trained for 100,000 epochs each. Results compared against the DeFoG paper (Table 7).

| Metric | Paper (DeFoG) | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|--------|---------------|--------|--------|--------|------------|
| Valid ↑ | 99.5 ± 1.0 | 99.0 | 97.7 | 99.5 | **98.7 ± 0.8** |
| Unique ↑ | 100.0 ± 0.0 | 100.0 | 100.0 | 100.0 | **100.0 ± 0.0** |
| Non-iso ↑ | 100.0 ± 0.0 | 100.0 | 100.0 | 100.0 | **100.0 ± 0.0** |
| Planar Acc ↑ | — | 99.0 | 97.7 | 99.5 | **98.7 ± 0.8** |
| Degree ↓ | 0.0005 ± 0.0002 | 0.000032 | 0.000038 | 0.000511 | **0.000194 ± 0.000221** |
| Spectre ↓ | 0.0072 ± 0.0011 | 0.004791 | 0.004527 | 0.004335 | **0.004551 ± 0.000189** |
| Clustering ↓ | 0.0501 ± 0.0149 | 0.020626 | 0.018320 | 0.037029 | **0.025325 ± 0.008415** |
| Orbit ↓ | 0.0006 ± 0.0004 | 0.000059 | 0.001910 | 0.000223 | **0.000731 ± 0.000824** |
| Wavelet ↓ | 0.0014 ± 0.0002 | 0.000016 | 0.000139 | 0.000167 | **0.000107 ± 0.000066** |

### Tree (3 seeds, completed)

Trained for 100,000 epochs each. Best checkpoint evaluated with 40 samples, 1000 denoising steps.

| Metric | Paper (DeFoG) | Seed 0 (best) | Seed 1 | Seed 2 | Mean ± Std |
|--------|---------------|---------------|--------|--------|------------|
| Valid ↑ | 100.0 | 100.0 | 95.0 | 100.0 | **98.3 ± 2.4** |
| Unique ↑ | 100.0 | 82.5 | 87.5 | 85.0 | **85.0 ± 2.0** |
| Non-iso ↑ | 100.0 | 100.0 | 100.0 | 100.0 | **100.0 ± 0.0** |
| Tree Acc ↑ | 100.0 | 100.0 | 95.0 | 100.0 | **98.3 ± 2.4** |
| Degree ↓ | — | 0.000575 | 0.000582 | 0.000287 | **0.000481 ± 0.000137** |
| Spectre ↓ | — | 0.010322 | 0.011042 | 0.011507 | **0.010957 ± 0.000488** |
| Clustering ↓ | — | 0.000000 | 0.000000 | 0.000000 | **0.000000 ± 0.000000** |
| Orbit ↓ | — | 0.000007 | 0.000013 | 0.000000 | **0.000007 ± 0.000005** |
| Wavelet ↓ | — | 0.000613 | 0.000518 | 0.000586 | **0.000572 ± 0.000039** |

*Training command (per seed):*
```bash
TL_BACKEND="torch" python defog_trainer.py \
  --dataset tree \
  --data_root ./datasets \
  --save_dir ./checkpoints_tree_seed${seed}_final \
  --seed ${seed} \
  --gpu 0 \
  --n_layers 10 --hidden_mlp_X 128 --hidden_mlp_E 64 --hidden_mlp_y 128 \
  --dx 256 --de 64 --dy 64 --dim_ffX 256 --dim_ffE 64 --dim_ffy 256 \
  --n_head 8 --n_epochs 100000 --batch_size 64 --lr 2e-4 \
  --train_distortion polydec --sample_distortion polydec \
  --omega 0 --eta 0 --sample_steps 1000 \
  --check_val_every_n_epochs 2000 --sample_every_val 1 --val_num_samples 40
```

### QM9 no-H (3 seeds, completed)

Trained for 1,000 epochs each without explicit hydrogens. Results compared against the DeFoG paper (no-H, 500 steps). **Best checkpoint evaluated with 10,000 samples, 500 denoising steps.**

| Metric | Paper (DeFoG) | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|--------|---------------|--------|--------|--------|------------|
| Validity ↑ | 99.3 ± 0.0 | 96.22 | 99.38 | 99.01 | **98.20 ± 1.41** |
| Relaxed Validity ↑ | 99.4 ± 0.1 | 97.19 | 99.57 | 99.18 | **98.65 ± 1.04** |
| Uniqueness ↑ | 96.3 ± 0.3 | 97.47 | 96.35 | 96.40 | **96.74 ± 0.52** |
| Novelty ↑ | — | 59.54 | 33.17 | 33.47 | **42.06 ± 12.36** |
| FCD ↓ | 0.12 ± 0.00 | 0.5782 | 0.1202 | 0.1033 | **0.2672 ± 0.2199** |

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
