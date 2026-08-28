# GraphTokenizer BERT Paper Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate and launch strict five-run GT+BERT paper experiments on QM9, OGBG-molhiv, and Peptides-struct without disturbing the active GT+GTE experiments.

**Architecture:** Enforce BERT-Small centrally in the official paper adapter, keep dataset training hyperparameters in three BERT-specific JSON files, and reuse the proven paper trainer and one-seed diagnostic wrapper. Gate formal tmux launch on code tests, preflight, complete-split diagnostic runs, checkpoint audits, and GPU telemetry.

**Tech Stack:** Python 3.10, PyTorch 2.1.2+cu121, Transformers 4.40.2, DGL 2.4.0+cu121, PyTorch Geometric 2.4.0, pytest, tmux, RTX 3090.

## Global Constraints

- Preserve active GTE sessions and their files.
- Use official dataset splits, Feuler, training-only BPE, and 2,000 merges.
- Store all generated environments, caches, temporary files, logs, checkpoints, and results under `/local/wrq`.
- Formal runs use seeds 42 through 46 and emit per-run plus mean/std task metrics.
- Do not launch formal BERT sessions unless all three complete-split diagnostic runs pass.

---

### Task 1: Enforce and report the exact BERT-Small architecture

**Files:**
- Modify: `examples/graph_tokenizer/official_model_adapter.py`
- Modify: `tests/models/test_graph_tokenizer_official_adapter.py`

**Interfaces:**
- Consumes: `create_paper_model(encoder_type, vocab_size, pad_token_id, task_type, output_dim, pooling, model_config)`.
- Produces: a BERT paper model whose `manifest()` records the exact BERT-Small configuration.

- [ ] Add a regression test that constructs the BERT paper model while passing conflicting architecture values and asserts 4 layers, hidden 512, 4 heads, FFN 2048, GELU, both dropouts 0.1, absolute positions, capacity 768, and LayerNorm epsilon `1e-12`.
- [ ] Run `python -m pytest -q tests/models/test_graph_tokenizer_official_adapter.py` and confirm the new test fails because the caller's maximum length is currently accepted and BERT manifest fields are absent.
- [ ] Add `PAPER_BERT_CONFIG` and apply it unconditionally to `BertConfig`; validate the constructed config and include the fields in `manifest()`.
- [ ] Rerun the adapter test file and require all tests to pass.

### Task 2: Add dataset-specific BERT paper configurations

**Files:**
- Create: `examples/graph_tokenizer/configs/qm9_bert_paper.json`
- Create: `examples/graph_tokenizer/configs/molhiv_bert_paper.json`
- Create: `examples/graph_tokenizer/configs/peptides_struct_bert_paper.json`
- Modify: `tests/models/test_graph_tokenizer_paper_protocol.py`

**Interfaces:**
- Consumes: `load_paper_preset(args, spec)` JSON override behavior.
- Produces: auditable BERT presets selected through `--paper-config`.

- [ ] Add a parameterized test loading all three JSON files and asserting encoder `bert`, Feuler, 200/200 epochs, MLM LR `1e-4`, dataset fine-tuning LR and batch size, all common optimizer settings, and maximum position capacity 768.
- [ ] Run the new test and confirm it fails because the files do not exist.
- [ ] Add the three exact JSON configurations.
- [ ] Rerun protocol tests and require all tests to pass.

### Task 3: Strict environment regression and preflight

**Files:**
- Read: the modified adapter, configurations, and GraphTokenizer tests.
- Write outputs under: `/local/wrq/graph-tokenizer-bert-paper-small-20260728/preflight/`.

**Interfaces:**
- Consumes: strict paper virtual environment and cached official datasets.
- Produces: passing pytest and three `status=ok` preflight logs.

- [ ] Run the complete relevant 82-test GraphTokenizer suite in `/local/wrq/gammagl-graphtokenizer-paper-cu121`.
- [ ] Run a real CUDA BERT forward pass and verify finite MLM logits and supervised output widths 16/1/11.
- [ ] Run three exact-command preflights with `--model bert`, BERT JSON configs, C++ BPE, five runs, and seed 42.
- [ ] Confirm runtime versions, dataset splits, and BPE backend have zero preflight errors.

### Task 4: Complete-split one-seed BERT diagnostics

**Files:**
- Create operational scripts under: `/local/wrq/graph-tokenizer-bert-paper-small-20260728/scripts/`.
- Write results under: `/local/wrq/graph-tokenizer-bert-paper-small-20260728/results/`.

**Interfaces:**
- Consumes: the existing one-seed diagnostic wrapper pattern and BERT configs overridden only to one MLM epoch, one fine-tuning epoch, and patience one.
- Produces: three diagnostic summaries and checkpoints.

- [ ] Generate small-run configs that preserve every formal parameter except epoch counts and patience.
- [ ] Launch QM9, MolHIV, and Peptides-struct diagnostics in separate tmux sessions on currently idle GPUs 0, 1, and 4.
- [ ] Record memory, utilization, temperature, and performance state every 30 seconds.
- [ ] Wait for all three exit files and require exit code zero.

### Task 5: Diagnostic audit and formal BERT launch

**Files:**
- Create operational scripts under: `/local/wrq/graph-tokenizer-bert-paper-formal-20260728/scripts/`.
- Write formal results under: `/local/wrq/graph-tokenizer-bert-paper-formal-20260728/results/`.

**Interfaces:**
- Consumes: passing diagnostic summaries/checkpoints and idle-GPU telemetry.
- Produces: three live five-seed BERT formal sessions plus one monitor session.

- [ ] Assert finite diagnostic metrics and model manifests with exact BERT-Small fields and output widths 16/1/11.
- [ ] Load checkpoints on CPU and assert every floating tensor is finite FP32.
- [ ] Compare complete-split mappings when an idle GPU thermally throttles; select the mapping with the lowest observed peak temperature while retaining driver thermal protection.
- [ ] Syntax-check and preflight formal scripts using 200/200 epochs, patience 20, runs five, and seeds 42–46.
- [ ] Launch three formal tmux sessions and the telemetry monitor.
- [ ] Verify three live training processes, nonzero GPU activity after preprocessing, no exit files, no error signatures, and exclusively `/local/wrq` cache bindings.
- [ ] Report tmux, process, log, telemetry, completion-status, and result-summary commands to the user.
