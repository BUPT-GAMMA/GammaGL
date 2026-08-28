# Native TensorLayerX GraphTokenizer Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the paper-only Hugging Face adapter with direct, native TensorLayerX `GraphBERT` and `GraphGTE` implementations while preserving the strict paper architecture and training interface.

**Architecture:** `graph_bert.py` provides native TLX BERT-Small plus shared attention/pooling helpers. `graph_gte.py` provides a separate TLX GTE encoder with RoPE, NTK scaling, and gated GELU MLP. `paper_protocol.py` loads these files directly and retains its PyTorch-only training loop.

**Tech Stack:** Python 3.10, TensorLayerX 0.5.8 public APIs, PyTorch 2.1.2 paper backend, pytest.

## Global Constraints

- Do not perform Git commits, branch operations, pushes, resets, or worktree operations.
- Model implementation files may import TensorLayerX and Python standard-library modules only; no direct PyTorch or Transformers imports.
- Preserve legacy dictionary output and paper `task="mlm"` / `task="supervised"` output contracts.
- BERT defaults: 4 layers, hidden 512, 4 heads, FFN 2048, absolute positions 768, GELU, dropout 0.1, LayerNorm epsilon `1e-12`.
- GTE defaults: 12 layers, hidden 768, 12 heads, intermediate 3072, RoPE length 8192, `rope_theta=20000`, NTK factor `8.0`, gated GELU MLP, dropout 0.1, LayerNorm epsilon `1e-12`.
- Formal validation uses the pinned environment `/local/wrq/gammagl-graphtokenizer-paper-cu121` and local caches.
- Do not interrupt adapter-version formal processes that were already running before this refactor.

---

### Task 1: Direct-model paper protocol contract

**Files:**
- Modify: `tests/models/test_graph_tokenizer_paper_protocol.py`
- Modify: `examples/graph_tokenizer/paper_protocol.py`

**Interfaces:**
- Produces: `_load_paper_model_classes() -> tuple[type, type]` and `require_paper_runtime(require_cuda=True)` in `paper_protocol.py`.
- Consumes later: `GraphBERT` and `GraphGTE` constructors from Tasks 2 and 3.

- [ ] Add a test that loads `paper_protocol.py`, calls `_load_paper_model_classes()`, and asserts returned class modules originate from `gammagl/models/graph_bert.py` and `graph_gte.py`.
- [ ] Add a test that sets `TL_BACKEND=tensorflow` and verifies `require_paper_runtime(require_cuda=False)` rejects strict paper execution.
- [ ] Run the two tests and verify they fail because the protocol currently exposes only `_load_adapter()`.
- [ ] Implement the direct dynamic loader and runtime validation with delayed TensorLayerX/PyTorch imports in the protocol.
- [ ] Run the tests and verify they pass.

### Task 2: Native TLX BERT-Small

**Files:**
- Modify: `tests/models/test_graph_transformer.py`
- Modify: `gammagl/models/graph_bert.py`

**Interfaces:**
- Produces: `GraphBERT(..., pad_token_id=0, task_type="regression", pooling="mean", layer_norm_eps=1e-12)`.
- Produces: `forward(input_ids, attention_mask=None, task=None)` and `manifest() -> dict`.
- Produces shared helpers for additive masks, masked mean pooling, parameter counts, head reshaping, and scaled dot-product attention.

- [ ] Add a small-model test for legacy dictionary output and paper task-specific outputs.
- [ ] Add a test asserting masked padding does not alter the pooled output when the padded token IDs change.
- [ ] Add a strict-default manifest test for BERT-Small architecture and position type.
- [ ] Add a gradient test requiring finite, nonzero gradients through attention, MLM head, and supervised head.
- [ ] Run these tests and verify failures identify the missing paper task API, manifest, and exact native architecture.
- [ ] Implement TLX embeddings, Q/K/V attention, additive masking, post-norm Transformer layers, GELU FFN, pooling, MLM head, task head, and manifest.
- [ ] Preserve old constructor parameters and `task=None` dictionary output.
- [ ] Run focused BERT tests until all pass.

### Task 3: Native TLX GTE with RoPE and gated MLP

**Files:**
- Modify: `tests/models/test_graph_transformer.py`
- Modify: `gammagl/models/graph_gte.py`

**Interfaces:**
- Produces: `GraphGTE(..., rope_theta=20000.0, rope_scaling_factor=8.0, position_embedding_type="rope")`.
- Produces: `RotaryEmbedding`, `apply_rotary_pos_emb`, a GTE attention block, gated GELU MLP, `forward(...)`, and `manifest()`.

- [ ] Add a hand-derived RoPE test using a fixed four-dimensional query/key fixture and literal cosine/sine expectations.
- [ ] Add a test proving the GTE embedding path has no learned absolute position embedding.
- [ ] Add a test that mutating the gate projection changes GTE output, protecting the gated MLP branch.
- [ ] Add strict-default manifest assertions for layers, dimensions, RoPE, theta, scaling, maximum length, and LayerNorm epsilon.
- [ ] Add small-model forward, padding-invariance, and finite-gradient tests.
- [ ] Run these tests and verify they fail against the current long-window BERT subclass.
- [ ] Implement native TLX rotary tables, NTK adjustment, rotary Q/K application, packed QKV attention, gated GELU MLP, post-norm GTE blocks, heads, and manifest.
- [ ] Run focused GTE tests until all pass.

### Task 4: Switch paper execution and delete adapter

**Files:**
- Modify: `examples/graph_tokenizer/paper_protocol.py`
- Modify: `tests/models/test_graph_tokenizer_paper_protocol.py`
- Modify: `tests/models/test_graph_transformer.py`
- Delete: `examples/graph_tokenizer/official_model_adapter.py`
- Delete: `tests/models/test_graph_tokenizer_official_adapter.py`

**Interfaces:**
- Consumes: direct `GraphBERT` and `GraphGTE` classes.
- Preserves: `_train_one_epoch`, `_evaluate`, checkpoint, early-stopping, and result-summary behavior.

- [ ] Add a paper-protocol test that exercises model construction for both encoder names using reduced architecture overrides intended only for tests.
- [ ] Run it and verify failure while `create_paper_model` is still adapter-based.
- [ ] Replace adapter creation with direct class selection and constructor calls.
- [ ] Validate all trainable model parameters are FP32 before AdamW creation in strict paper mode.
- [ ] Migrate behavior assertions from the adapter test into model/protocol tests.
- [ ] Delete the adapter and adapter-only test with `apply_patch`.
- [ ] Run a repository text search confirming no runtime or test reference remains.
- [ ] Run focused model and protocol tests.

### Task 5: Regression and local smoke verification

**Files:**
- Modify: `examples/graph_tokenizer/README.md` only if it mentions adapter/Hugging Face model construction.
- Verify: GraphTokenizer source and tests.

**Interfaces:**
- Consumes: existing CLI preflight and smoke modes.
- Produces: local test evidence and smoke artifacts outside the home directory.

- [ ] Run Python syntax compilation for both model files and `paper_protocol.py`.
- [ ] Run `tests/models/test_graph_transformer.py`, `test_graph_tokenizer_paper_protocol.py`, and all GraphTokenizer transform/model tests.
- [ ] Run BERT and GTE `--preflight` commands in the pinned environment with cache roots under `/local/wrq`.
- [ ] Run one tiny BERT paper-protocol test and one tiny GTE paper-protocol test on an available GPU only after checking that doing so will not disturb formal jobs.
- [ ] Scan smoke logs for traceback, OOM, runtime errors, NaN, non-finite gradients, killed processes, and missing checkpoints.
- [ ] Verify the smoke result manifests report native `GraphBERT` / `GraphGTE` architecture fields.
- [ ] Report any validation that cannot safely run while old formal jobs are active instead of interrupting them.
