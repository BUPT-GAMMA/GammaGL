# QM9 dual-scale evaluation

## Goal

Make QM9's paper MAE use the same train-split standardized target space as
training, validation selection, and early stopping. Preserve a separately
named raw-unit diagnostic so that large-energy targets remain inspectable.

## Scope

Only the paper-protocol evaluation and its emitted result fields change.
Model architecture, tokenizer, label standardization, optimization, splits,
and Peptides-struct/MolHIV metric semantics remain unchanged.

## Evaluation contract

For QM9, `_evaluate` will keep the concatenated normalized labels and
predictions as the primary evaluation tensors.

- `metric` is the all-elements MAE in normalized space.
- `metric_space` is `"train_standardized"`.
- `per_target_mae` is the normalized MAE for each QM9 target.
- `per_target_mae_raw` is computed from separately denormalized copies.

For Peptides-struct, `metric` and `per_target_mae` remain raw-unit Average
MAE diagnostics. MolHIV continues to use sigmoid probabilities with the OGB
ROC-AUC evaluator.

Because finetuning validation calls `_evaluate`, the corrected QM9 metric is
also the value used to save the best checkpoint and advance early stopping.

## Result aggregation

The existing primary metric field remains `mae`; it now means normalized QM9
MAE. The paper summary retains aggregation of `per_target_mae` and adds
aggregation of `per_target_mae_raw` when present. Result metadata identifies
the primary metric space so reports cannot silently mix units.

## Verification

Tests will first demonstrate that, for a two-target QM9 normalizer with a
large second-target standard deviation, the primary MAE remains the
normalized value while `per_target_mae_raw` changes by the standard
deviation. A Peptides regression test will prove that its raw-unit behavior
is unchanged. A checkpoint-only evaluation of the existing BERT-QM9 run will
then verify a result near 0.117 rather than 20.96, without retraining.

Existing finished QM9 checkpoints remain useful for metric validation, but
they are not strict final-paper checkpoints because their historical best
epoch was selected with the old raw-unit metric. A subsequent QM9 training
run is required for a formal result under the corrected protocol.
