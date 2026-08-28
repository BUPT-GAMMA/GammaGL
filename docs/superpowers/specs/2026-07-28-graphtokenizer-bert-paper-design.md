# GraphTokenizer BERT Paper Runs Design

## Goal

Run GraphTokenizer GT+BERT on QM9, OGBG-molhiv, and Peptides-struct with the same strict data, tokenizer, evaluation, five-run aggregation, environment, and local-storage guarantees already validated for GT+GTE.

## Requirements

- Use the official dataset splits and train BPE only on each training split.
- Use Feuler serialization and 2,000 BPE merges for all datasets.
- Jointly regress all 16 QM9 targets; use ROC-AUC for MolHIV and Average MAE for Peptides-struct.
- Use BERT-Small: 4 layers, hidden size 512, 4 attention heads, FFN size 2048, GELU, dropout 0.1, learned absolute position embeddings, maximum position capacity 768, and LayerNorm epsilon `1e-12`.
- Use MLM learning rate `1e-4` on all three datasets. Fine-tuning learning rates are `1e-5`, `5e-5`, and `1e-5` for QM9, MolHIV, and Peptides-struct respectively.
- Preserve all other paper settings: 200 MLM epochs, up to 200 fine-tuning epochs, AdamW, weight decay 0.1, warmup ratios 0.12/0.025, gradient norms 2.0/0.5, mask probability 0.09, patience 20, and batches 32/32/16.
- Run five seeds `42,43,44,45,46` and report mean and standard deviation.
- Keep environments, caches, temporary files, logs, checkpoints, and results under `/local/wrq`.
- Do not interrupt or alter the active GTE formal runs.

## Design

### Configuration isolation

Add one BERT-specific JSON configuration per dataset. Do not reuse or mutate the GTE JSON files because the GTE and BERT MLM learning rates and maximum position capacities differ. The trainer continues to consume the same `--paper-config` interface.

### Exact architecture enforcement

Define the BERT-Small paper architecture in `official_model_adapter.py` and apply it unconditionally whenever `encoder_type="bert"`. The current adapter accepts a caller-provided effective sequence length and would otherwise create smaller learned position tables (for example 14 positions on QM9), which violates the requested 768-position architecture and changes parameter count. Validate the constructed configuration and expose its fields in the model manifest, just as the GTE path does.

### Test gate

Use test-driven development for the architecture correction and configuration files. Run the complete GraphTokenizer regression suite in the strict paper environment, then run three exact-command preflights. Run one complete official-split seed with one MLM epoch and one fine-tuning epoch for each dataset. Require exit code zero, finite task metrics, exact model manifest, expected output widths 16/1/11, finite FP32 checkpoints, no log error signatures, and acceptable memory/thermal telemetry.

### Formal execution

Only after all three diagnostic runs pass, launch three independent tmux sessions on GPUs that are idle at launch. Keep a fourth session for 30-second GPU telemetry. If a selected GPU enters thermal slowdown during the diagnostic run, compare a second complete-split mapping. If every available three-card mapping is driver-throttled, choose the mapping with the lowest observed peak temperature and disclose that the throttle affects runtime rather than metric semantics. Formal result summaries must contain per-seed values plus mean and standard deviation for MAE, ROC-AUC, and Average MAE.

## Alternatives considered

1. **Independent BERT configurations plus adapter enforcement (selected):** reproducible and prevents parameter leakage between GTE and BERT runs.
2. **Change the global dataset presets to nested per-model presets:** centralizes values but is a larger, riskier refactor while GTE jobs are active.
3. **Override everything only on the command line:** avoids repository files but is hard to audit and reproduce and can silently retain GTE values.

## Failure handling

Any preflight, regression, OOM, non-finite value, or checkpoint mismatch blocks formal BERT launch. Thermal slowdown blocks the initial mapping and requires the comparison procedure above; it blocks the overall launch only if the driver cannot keep temperature within its protected operating range. Diagnose and regression-test software defects before rerunning. Preserve failed logs and telemetry under the diagnostic output root.
