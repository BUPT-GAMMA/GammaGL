# GraphTokenizer GT+GTE Strict Paper Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the paper protocol, validate an exact CUDA 12.1 runtime, pass full-split one-epoch tests, and launch five-run GT+GTE benchmarks for QM9, OGBG-molhiv and Peptides-struct on three GPUs.

**Architecture:** Keep changes inside existing GraphTokenizer implementation, tests, configs and README. Create the exact software environment and all runtime artifacts under `/local/wrq`; use strict preflight and full-split small runs as hard gates before three independent tmux formal jobs.

**Tech Stack:** Python 3.10, PyTorch 2.1.2+cu121, DGL 2.4.0+cu121, PyTorch Geometric 2.4.0, Transformers, OGB, TensorLayerX, C++ GraphBPE, pytest, tmux.

## Global Constraints

- Run only GT+GTE for QM9, OGBG-molhiv and Peptides-struct.
- Keep the original GammaGL repository structure and public APIs unchanged.
- Preserve all existing user changes; do not reset, clean, commit or modify unrelated files.
- Store every new environment, cache, temporary file, checkpoint, log and result under `/local/wrq`.
- Use official split indices and fit Feuler/BPE only on train, with exactly 2000 merges.
- Use five formal seeds 42, 43, 44, 45 and 46; report mean and population standard deviation.
- Do not launch formal tmux jobs until all unit tests, preflight checks and three full-split small runs pass.

---

### Task 1: QM9 16-Target Paper Contract

**Files:**
- Modify: `tests/models/test_graph_tokenizer_paper_protocol.py`
- Modify: `examples/graph_tokenizer/paper_protocol.py`
- Modify: `examples/graph_tokenizer/graph_tokenizer_trainer.py`

**Interfaces:**
- Produces: `_prepare_labels(encoded, spec, target_property=None)` returning vector `mean`/`std` and `output_dim=16` for QM9.
- Produces: `_denormalize_labels(torch, values, normalizer)` with device/dtype-safe vector broadcasting.
- Changes: strict paper QM9 no longer requires `--target-property`; a supplied property is rejected to avoid silently running the old single-target protocol.

- [ ] **Step 1: Write failing tests for the 16-target contract.**

  Add tests that pass two-target hand-checked rows `[[1, 10], [3, 14]]`, assert train mean `[2, 12]`, std `[1, 2]`, normalized rows `[[-1, -1], [1, 1]]`, unchanged 2-D output, and restored values. Add a test that `validate_paper_args` accepts QM9 without `target_property` and rejects a supplied single target.

- [ ] **Step 2: Run the focused tests and confirm RED.**

  Run: `TL_BACKEND=torch /local/wrq/gammagl-graphtokenizer/bin/python -m pytest tests/models/test_graph_tokenizer_paper_protocol.py -q`

  Expected: failures show the current scalar target lookup and scalar normalizer behavior.

- [ ] **Step 3: Implement the minimal multi-target behavior.**

  Compute each column's train-only mean and population standard deviation, reject non-finite labels and wrong widths, normalize all splits column-wise, and return the dataset output dimension. Convert the stored vectors to Torch tensors inside evaluation before restoring units.

- [ ] **Step 4: Run the focused tests and confirm GREEN.**

  Run the same pytest command; expected all protocol tests pass.

### Task 2: Exact Feuler And GTE Architecture

**Files:**
- Modify: `tests/models/test_graph_tokenizer_official_adapter.py`
- Modify: `tests/models/test_graph_tokenizer_paper_protocol.py`
- Modify: `examples/graph_tokenizer/official_model_adapter.py`
- Modify: `examples/graph_tokenizer/paper_protocol.py`
- Modify: `examples/graph_tokenizer/configs/qm9_paper.json`
- Modify: `examples/graph_tokenizer/configs/molhiv_paper.json`
- Modify: `examples/graph_tokenizer/configs/peptides_struct_paper.json`

**Interfaces:**
- Produces: GTE configuration validation with exact layers/width/heads/FFN/activation/dropouts/RoPE/max-length/epsilon.
- Produces: model manifest fields for the full architecture and total/trainable parameter counts.
- Changes: all three presets resolve to `serialization="feuler"` and `max_position_embeddings=8192`.

- [ ] **Step 1: Write failing architecture and preset tests.**

  Use fake AutoConfig/AutoModel classes to assert the adapter passes attention dropout 0.1, hidden dropout 0.1, RoPE and all required dimensions into model construction. Assert each dataset preset resolves to Feuler and 8192.

- [ ] **Step 2: Run the focused tests and confirm RED.**

  Run: `TL_BACKEND=torch /local/wrq/gammagl-graphtokenizer/bin/python -m pytest tests/models/test_graph_tokenizer_official_adapter.py tests/models/test_graph_tokenizer_paper_protocol.py -q`

  Expected: MolHIV/Peptides fail on Eulerian, QM9 fails on 8096, and the GTE fake observes attention dropout 0.0 or missing overrides.

- [ ] **Step 3: Implement exact settings and manifest validation.**

  Override the released config before `AutoModel.from_config`, verify resulting values, instantiate the model, and expose the exact configuration in `manifest()`. Keep FP32 model promotion and the existing stable MLM masking fix.

- [ ] **Step 4: Run focused tests and confirm GREEN.**

  Run the same combined pytest command; expected all pass.

### Task 3: Exact Local Runtime

**Files:**
- Create runtime prefix: `/local/wrq/gammagl-graphtokenizer-paper-cu121`
- Create caches/temp: `/local/wrq/cache/graphtokenizer-paper`, `/local/wrq/tmp/graphtokenizer-paper`

**Interfaces:**
- Produces: Python interpreter importing `torch==2.1.2+cu121`, `dgl==2.4.0+cu121`, `torch_geometric==2.4.0` and CUDA.

- [ ] **Step 1: Create the Python 3.10 environment under `/local`.**

  Run: `/local/wrq/gammagl-graphtokenizer/bin/python -m venv /local/wrq/gammagl-graphtokenizer-paper-cu121`

- [ ] **Step 2: Install exact framework versions and compatible protocol dependencies.**

  Run PyTorch installation from `https://download.pytorch.org/whl/cu121`, DGL from `https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html`, and install `torch-geometric==2.4.0`, the existing compatible TensorLayerX/Transformers/OGB/NumPy/pybind11 dependencies without writing pip cache to home.

- [ ] **Step 3: Audit exact imports and CUDA execution.**

  Execute a Python audit that asserts exact versions, `torch.version.cuda == "12.1"`, CUDA availability, RTX 3090 visibility, and one CUDA tensor operation.

- [ ] **Step 4: Build and import C++ GraphBPE with the exact interpreter.**

  Run: `TL_BACKEND=torch /local/wrq/gammagl-graphtokenizer-paper-cu121/bin/python third_party/graph_bpe_cpp/setup.py build_ext --inplace`

  Then import `third_party.graph_bpe_cpp` and assert `is_available()`.

### Task 4: Regression Suite And Official Data Preflight

**Files:**
- Modify: `examples/graph_tokenizer/README.md`
- Runtime reports: `/local/wrq/graph-tokenizer-paper-small-20260728/preflight/`

**Interfaces:**
- Produces: passing GraphTokenizer regression suite and machine-readable runtime/data audit for all three datasets.

- [ ] **Step 1: Run the complete relevant test suite in the exact environment.**

  Run the dataset, tokenizer, C++ backend, adapter and paper-protocol test files with `TL_BACKEND=torch`; require zero failures.

- [ ] **Step 2: Run paper preflight for all datasets.**

  Use `/local/wrq/graph-tokenizer-data`, C++ BPE and GTE. Verify total/split counts, split disjointness, task dimensions, label finiteness/NaN rules, train-only tokenizer fitting and model architecture.

- [ ] **Step 3: Record all resolved settings.**

  Save package versions, GPU/driver, Git revision/dirty state, dataset source/hash, split sizes, seeds and config JSON beneath the local preflight directory.

- [ ] **Step 4: Update README commands.**

  Replace single-target QM9 and old environment examples with the exact local-cache, small-test and formal GT+GTE commands.

### Task 5: Three Full-Split Small Runs

**Files:**
- Create runtime configs/logs/results under `/local/wrq/graph-tokenizer-paper-small-20260728`

**Interfaces:**
- Consumes: exact runtime, corrected protocol and official splits.
- Produces: one finite metric and restorable FP32 checkpoint for each dataset after 1 MLM epoch and 1 fine-tuning epoch.

- [ ] **Step 1: Create diagnostic overrides without weakening formal configs.**

  Write three `/local` JSON overrides that retain formal encoder, Feuler, batch size, learning rates, warmups, gradient norms, mask probability, 2000 BPE merges and max length, changing only runs/MLM epochs/fine-tune epochs to one through the existing diagnostic runner.

- [ ] **Step 2: Launch the three small runs on GPUs 3, 4 and 5.**

  Export local `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME`, `TMPDIR`, `PIP_CACHE_DIR` and `CUDA_VISIBLE_DEVICES`; tee each process to its own log and status file.

- [ ] **Step 3: Monitor safety and completion.**

  Sample GPU memory, utilization and temperature. Fail the gate on nonzero status, OOM, non-finite value, missing metric/checkpoint, wrong split/config/model manifest, or checkpoint tensors outside FP32/non-finite.

- [ ] **Step 4: Summarize the gate.**

  Record QM9 MAE, MolHIV ROC-AUC, Peptides Average MAE, peak VRAM and elapsed time. Proceed only if all three pass.

### Task 6: Formal Three-GPU tmux Benchmark

**Files:**
- Create runtime launch scripts/logs/results under `/local/wrq/graph-tokenizer-paper-formal-20260728`

**Interfaces:**
- Produces: tmux sessions `gt_gte_qm9_formal`, `gt_gte_molhiv_formal`, `gt_gte_pstruct_formal`.
- Produces: five per-seed test metrics and mean/population-std summaries for each dataset.

- [ ] **Step 1: Generate immutable resolved formal configs and launch scripts.**

  Use 200 MLM epochs, maximum 200 fine-tune epochs, patience 20, paper batch sizes/LRs/warmups/gradient norms/weight decay/mask probability, Feuler, GTE and seeds 42–46. Each script writes PID, start/end time and exit status.

- [ ] **Step 2: Validate launch commands without training.**

  Run shell syntax checks and trainer preflight using each resolved config. Confirm all paths are under `/local/wrq` except the read-only repository source.

- [ ] **Step 3: Launch three tmux sessions on GPUs 3, 4 and 5.**

  Start one dataset per GPU and immediately inspect the first log lines, process tree and GPU allocation. Do not reuse stale session names.

- [ ] **Step 4: Report operational commands.**

  Provide `tmux ls`, `tmux attach -t ...`, `tail -f ...`, `nvidia-smi`, `ps` and JSON result inspection commands with exact paths.

## Plan Self-Review

- Every user-specified GT+GTE architecture, tokenization, split, training and metric requirement maps to Tasks 1–6.
- Unknown paper choices are explicit: seeds are 42–46, scheduler is linear warmup/decay, mean/std uses `ddof=0`, and sequences over 8192 fail rather than being silently truncated.
- TDD precedes every source behavior change; environment/config-only work is validated by executable audits and preflight.
- No step modifies repository layout, unrelated dirty files or home-directory caches.
- Formal launch is mechanically blocked until the three full-split small runs pass.
