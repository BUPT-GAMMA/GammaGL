# QM9 Dual-Scale Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report QM9 paper MAE and select checkpoints in train-standardized target space while retaining explicit raw-unit per-target diagnostics.

**Architecture:** `_evaluate` will compute the QM9 primary metric from the normalized tensors already used by training. It will separately denormalize copies only to produce `per_target_mae_raw`. The trainer will preserve `metric_space` and aggregate both normalized and raw per-target MAE fields without changing MolHIV or Peptides behavior.

**Tech Stack:** Python 3.10, PyTorch 2.1.2, TensorLayerX/PyTorch backend, pytest.

## Global Constraints

- Keep model architecture, tokenizer, data splits, label normalization, optimizer, and loss unchanged.
- QM9 `metric` and checkpoint selection use `train_standardized` MAE.
- QM9 emits `per_target_mae` in standardized space and `per_target_mae_raw` in raw units.
- Peptides-struct remains raw-unit Average MAE; MolHIV remains probability-space OGB ROC-AUC.
- Do not modify or terminate the running `gt20260731_formal_gte_qm9` process.
- Do not run Git commands or create Git commits.

---

### Task 1: Establish red regression tests for evaluation spaces

**Files:**
- Modify: `tests/models/test_graph_tokenizer_paper_protocol.py`
- Modify: `tests/models/test_graph_transformer.py`

**Interfaces:**
- Consumes: `paper_protocol._evaluate(torch, model, loader, spec, device, normalizer)`.
- Produces: an evaluation result with `metric`, `metric_space`, `per_target_mae`, and optional `per_target_mae_raw`.
- Consumes: `graph_tokenizer_trainer.aggregate_per_target_mae(summaries, field_name="per_target_mae")`.
- Produces: per-split mean and population standard deviation for the requested field.

- [ ] **Step 1: Write the failing QM9 dual-scale evaluation test**

```python
def test_qm9_evaluation_uses_standardized_metric_and_retains_raw_diagnostics():
    protocol = _load_paper_protocol_module()

    class FixedModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask, task):
            return torch.tensor([[1.0, 1.0]])

    loader = [(torch.tensor([[1]]), torch.tensor([[1]]),
               torch.tensor([[0.0, 0.0]]))]
    spec = SimpleNamespace(canonical_name="qm9", label_keys=("small", "large"))
    result = protocol._evaluate(
        torch, FixedModel(), loader, spec, torch.device("cpu"),
        normalizer={"mean": [0.0, 0.0], "std": [1.0, 100.0]},
    )

    assert result["metric"] == 1.0
    assert result["metric_space"] == "train_standardized"
    assert result["per_target_mae"] == {"small": 1.0, "large": 1.0}
    assert result["per_target_mae_raw"] == {"small": 1.0, "large": 100.0}
```

- [ ] **Step 2: Write the failing unchanged-Peptides test**

```python
def test_peptides_evaluation_keeps_raw_metric_space():
    protocol = _load_paper_protocol_module()

    class FixedModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask, task):
            return torch.tensor([[1.0, 3.0]])

    loader = [(torch.tensor([[1]]), torch.tensor([[1]]),
               torch.tensor([[0.0, 1.0]]))]
    spec = SimpleNamespace(canonical_name="peptides-struct", label_keys=("a", "b"))
    result = protocol._evaluate(torch, FixedModel(), loader, spec, "cpu", None)

    assert result["metric"] == 1.5
    assert result["metric_space"] == "raw"
    assert result["per_target_mae"] == {"a": 1.0, "b": 2.0}
    assert "per_target_mae_raw" not in result
```

- [ ] **Step 3: Write the failing raw-field aggregation test**

```python
def test_aggregate_per_target_mae_can_select_raw_field():
    trainer = _load_trainer_module()
    summaries = [{
        "best_val": {"per_target_mae_raw": {"energy": 10.0}},
        "best_test": {"per_target_mae_raw": {"energy": 20.0}},
    }]

    assert trainer.aggregate_per_target_mae(
        summaries, field_name="per_target_mae_raw") == {
            "val": {"energy": {"mean": 10.0, "std": 0.0}},
            "test": {"energy": {"mean": 20.0, "std": 0.0}},
        }
```

- [ ] **Step 4: Run only the new tests and verify RED**

Run:

```bash
TL_BACKEND=torch PYTHONPATH=/home/wrq/GammaGL \
/local/wrq/gammagl-graphtokenizer-paper-cu121/bin/python -m pytest \
tests/models/test_graph_tokenizer_paper_protocol.py \
tests/models/test_graph_transformer.py -k 'dual_scale or keeps_raw_metric_space or select_raw_field' -v
```

Expected: the QM9 assertion fails because current `_evaluate` reports raw-unit MAE; the aggregation test fails because `field_name` is not accepted.

### Task 2: Implement dual-space evaluation and result propagation

**Files:**
- Modify: `examples/graph_tokenizer/paper_protocol.py:874-920`
- Modify: `examples/graph_tokenizer/graph_tokenizer_trainer.py:636-659`
- Modify: `examples/graph_tokenizer/graph_tokenizer_trainer.py:1700-1741`
- Test: `tests/models/test_graph_tokenizer_paper_protocol.py`
- Test: `tests/models/test_graph_transformer.py`

**Interfaces:**
- `_evaluate` returns `dict` containing `loss`, `metric`, `metric_space`, `per_target_mae`, and for QM9 `per_target_mae_raw`.
- `aggregate_per_target_mae(summaries, field_name="per_target_mae")` aggregates the supplied per-target dictionary.
- Paper result summaries contain aggregate `metric_space`, `per_target_mae`, and `per_target_mae_raw` when available.

- [ ] **Step 1: Preserve normalized tensors as QM9 primary metric inputs**

Replace the single unconditional normalizer block in `_evaluate` with this behavior:

```python
    y_true = torch.cat(targets, dim=0)
    y_pred = torch.cat(predictions, dim=0)
    metric_space = "raw"
    raw_per_target_mae = None
    metric_y_true, metric_y_pred = y_true, y_pred
    if normalizer is not None:
        raw_y_true = _denormalize_labels(torch, y_true, normalizer)
        raw_y_pred = _denormalize_labels(torch, y_pred, normalizer)
        if spec.canonical_name == "qm9":
            metric_space = "train_standardized"
            raw_per_target_mae = compute_paper_metric_details(
                spec, raw_y_true.numpy(), raw_y_pred.numpy()
            )["per_target_mae"]
        else:
            metric_y_true, metric_y_pred = raw_y_true, raw_y_pred
```

Compute `metric_details` from `metric_y_true` and `metric_y_pred`; add
`metric_space` to the returned result and add `per_target_mae_raw` only when
`raw_per_target_mae` is non-`None`.

- [ ] **Step 2: Generalize per-target aggregation**

Change the helper signature and its dictionary reads:

```python
def aggregate_per_target_mae(
        summaries: Sequence[Dict[str, Any]],
        field_name: str = "per_target_mae") -> Dict[str, Any]:
    # Read summary[split_name].get(field_name, {}) for names and values.
```

Keep the default to preserve existing callers.

- [ ] **Step 3: Propagate metric-space and raw diagnostics through paper summaries**

When building each `best_val` and `best_test` in `run_paper_experiments`, copy:

```python
"metric_space": run["best_val"].get("metric_space", "raw"),
"per_target_mae_raw": run["best_val"].get("per_target_mae_raw", {}),
```

and the corresponding test fields. In the aggregate dictionary add:

```python
"metric_space": runs[0]["best_test"]["metric_space"] if runs else "raw",
"per_target_mae_raw": aggregate_per_target_mae(
    runs, field_name="per_target_mae_raw"),
```

- [ ] **Step 4: Run the red tests and verify GREEN**

Run the Task 1 command again.

Expected: all three new tests pass.

- [ ] **Step 5: Run the focused paper-protocol regression suite**

Run:

```bash
TL_BACKEND=torch PYTHONPATH=/home/wrq/GammaGL \
/local/wrq/gammagl-graphtokenizer-paper-cu121/bin/python -m pytest \
tests/models/test_graph_tokenizer_paper_protocol.py \
tests/models/test_graph_transformer.py -v
```

Expected: PASS; in particular, the existing FP32-before-denormalization test
still observes two FP32 denormalization calls for QM9 diagnostics.

### Task 3: Verify an existing checkpoint without retraining

**Files:**
- Use: `/local/wrq/graph-tokenizer-audit-20260804/qm9_checkpoint_audit.py`
- Output: `/local/wrq/graph-tokenizer-audit-20260804/qm9_checkpoint_audit_run0.json`

**Interfaces:**
- Consumes the existing BERT-QM9 run-0 checkpoint, encoded cache, and train normalizer.
- Produces independent normalized and raw MAE values from identical predictions.

- [ ] **Step 1: Run the checkpoint-only diagnostic on CPU**

Run:

```bash
TL_BACKEND=torch PYTHONPATH=/home/wrq/GammaGL \
/local/wrq/gammagl-graphtokenizer-paper-cu121/bin/python \
/local/wrq/graph-tokenizer-audit-20260804/qm9_checkpoint_audit.py
```

Expected: normalized MAE near `0.1173`; raw MAE near `20.95`; finite output;
label reconstruction error below `3e-4`.

- [ ] **Step 2: Record the formal-training consequence**

State in the result handoff that historical QM9 checkpoints used raw-unit
validation MAE for selection. Therefore the checkpoint-only result validates
the metric repair but does not replace a new formal QM9 run under corrected
early stopping.
