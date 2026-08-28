# GraphTokenizer Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GraphBPE, benchmark datasets, and ordinary token batching each have one reusable implementation.

**Architecture:** `GraphBPE` owns the Python algorithm; the optional third-party package exposes native C++ only. The trainer consumes GammaGL datasets and `GraphTokenizer` batch helpers, while the strict paper protocol retains its Torch-native dynamic masking semantics.

**Tech Stack:** Python, TensorLayerX, PyTorch, pytest, optional pybind11 C++ extension.

**Spec:** User-approved four-point reuse plan from this conversation.

## Global Constraints

- Use `/local/wrq/gammagl-graphtokenizer-paper-cu121/bin/python` with `TL_BACKEND=torch` for validation.
- Do not add dependencies or a generic trainer/tokenizer framework.
- Explicit `backend="cpp"` must fail when the native extension is unavailable; `backend="auto"` must use Python then.
- Preserve Torch dynamic MLM masking in `paper_protocol.py`.

---

### Task 1: BPE backend routing

**Files:**
- Modify: `gammagl/transforms/graph_bpe.py`
- Modify: `third_party/graph_bpe_cpp/__init__.py`
- Test: `tests/transforms/test_graph_tokenizer.py`

- [ ] Add failing tests for auto fallback and explicit cpp failure without a native extension.
- [ ] Make `GraphBPE` select the native bridge only when available and retain its own Python implementation otherwise.
- [ ] Remove Python BPE training/encoding fallback from the bridge and rerun BPE tests.

### Task 2: Shared ordinary MLM batching

**Files:**
- Modify: `gammagl/transforms/graph_tokenizer.py`
- Modify: `examples/graph_tokenizer/graph_tokenizer_trainer.py`
- Test: `tests/transforms/test_graph_tokenizer.py`
- Test: `tests/models/test_graph_transformer.py`

- [ ] Add failing tests for sequence padding and deterministic MLM batching exposed by `GraphTokenizer`.
- [ ] Reuse those methods from the ordinary trainer and smoke path.
- [ ] Keep `paper_protocol.py` Torch masking unchanged and rerun focused tests.

### Task 3: Dataset-only benchmark loading

**Files:**
- Modify: `examples/graph_tokenizer/graph_tokenizer_trainer.py`
- Test: `tests/models/test_graph_transformer.py`

- [ ] Add a failing test that benchmark loading requires the public GammaGL dataset path instead of raw pickle fallback.
- [ ] Remove trainer-local pickle, directory scanning, and split parsing helpers.
- [ ] Verify aliases and synthetic smoke behavior remain intact.

### Task 4: Strict-environment verification

**Files:**
- Test: `tests/transforms/test_graph_tokenizer.py`
- Test: `tests/datasets/test_graph_tokenizer_dataset_download.py`
- Test: `tests/models/test_graph_transformer.py`
- Test: `tests/models/test_graph_tokenizer_paper_protocol.py`

- [ ] Run focused unit suites and inspect failures.
- [ ] Run the full GraphTokenizer regression suites in the pinned environment.
