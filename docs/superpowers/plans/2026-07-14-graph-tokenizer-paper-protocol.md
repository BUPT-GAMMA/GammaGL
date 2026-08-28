# GraphTokenizer Paper-Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a Torch-only GraphTokenizer paper protocol for QM9, OGBG-molhiv and Peptides-struct without breaking GammaGL's existing multi-backend APIs.

**Architecture:** Keep GammaGL graph data and transforms backend-neutral. Add strict dataset conversion and artifact validation, then add an experiment-only Torch/HuggingFace adapter selected by `--protocol paper`. The paper path serializes graphs, fits BPE only on train, performs MLM and fine-tuning with paper presets, restores the best validation checkpoint, and evaluates test once.

**Tech Stack:** Python 3.10+, TensorLayerX, PyTorch, HuggingFace Transformers, OGB, RDKit, NumPy, optional PyG/LRGB conversion helper, pybind11 C++ extension, pytest.

## Global Constraints

- Keep `gammagl.models` importable under every existing TensorLayerX backend.
- `torch`, `transformers` and `ogb` are required only for `--protocol paper` and must be imported lazily.
- Preserve existing user changes and do not create commits, push branches, rebase or reset.
- Reject invalid data and splits; do not pad, truncate, substitute zeros or silently discard records.
- Paper runs require `TL_BACKEND=torch`, CUDA, five sequential seeds starting from 42 and fixed special-token IDs 0 through 7.

---

### Task 1: Strict Molecular Dataset Contract

**Files:**
- Modify: `gammagl/datasets/_molecular_benchmark.py`
- Modify: `gammagl/datasets/qm9.py`
- Modify: `gammagl/datasets/ogbg_molhiv.py`
- Modify: `gammagl/datasets/peptides_struct.py`
- Modify: `tests/datasets/test_molecule_benchmark_datasets.py`

**Interfaces:**
- Produces `validate_molecular_splits(num_samples, split_indices)` that raises `ValueError` for out-of-range, negative, duplicate or overlapping indices.
- Produces dataset-specific `extract_node_type_ids(graph_data)` and `extract_edge_type_ids(graph_data)` that reproduce the paper's atom/bond token rules.
- Produces `validate_label(label, num_tasks, allow_nan)` that preserves legal NaN and rejects incorrect label shapes.

- [ ] **Step 1: Write failing tests for invalid split, missing feature field, paper atom/bond feature extraction and short QM9 label.**

```python
with pytest.raises(ValueError, match="overlap"):
    validate_molecular_splits(3, {"train": [0], "val": [0], "test": [2]})
with pytest.raises(ValueError, match="exactly 16"):
    QM9(root=str(tmp_path))
assert dataset[0].x.tolist() == [13, 17]
```

- [ ] **Step 2: Run the focused dataset tests and verify they fail because the validator and row encoder do not exist.**

Run: `TL_BACKEND=torch pytest tests/datasets/test_molecule_benchmark_datasets.py -q`

Expected: failure identifying the current zero-padding and unconstrained feature-extraction behavior.

- [ ] **Step 3: Implement strict validation and dataset-specific paper feature extraction.**

```python
def validate_molecular_splits(num_samples, splits):
    seen = set()
    for name in ("train", "val", "test"):
        for index in splits[name]:
            if index < 0 or index >= num_samples or index in seen:
                raise ValueError(f"Invalid {name} split index: {index}")
            seen.add(index)
```

Implement data-source adapters in each dataset class, require their expected task dimensions, and convert atomic numbers to `2 * Z + 1` and bond types to `2 * bond_type`. Reject raw formats that do not expose the paper-required fields.

- [ ] **Step 4: Run the focused dataset tests and verify they pass.**

Run: `TL_BACKEND=torch pytest tests/datasets/test_molecule_benchmark_datasets.py -q`

Expected: all dataset tests pass, including strict rejection cases.

### Task 2: Reversible Deterministic Serialization And BPE Boundaries

**Files:**
- Modify: `gammagl/transforms/graph_serializer.py`
- Modify: `gammagl/transforms/graph_bpe.py`
- Modify: `gammagl/transforms/graph_tokenizer.py`
- Modify: `tests/transforms/test_graph_tokenizer.py`

**Interfaces:**
- Produces `FrequencyGuidedEulerianSerializer.deserialize(tokens)` returning `Graph`.
- Produces `GraphTokenizer.assert_round_trip(graph)` and `GraphTokenizer.max_token_id`.
- Produces `GraphBPE.fit(serialized_train_sequences)` without observing validation or test sequences.

- [ ] **Step 1: Write failing tests for serialize/decode graph equality, node-permutation equality and model vocabulary overflow.**

```python
tokens = serializer.serialize(graph)
assert graphs_equal(serializer.deserialize(tokens), graph)
assert tokenizer.encode(permuted_graph) == tokenizer.encode(graph)
with pytest.raises(ValueError, match="vocab_size"):
    tokenizer.validate_model_vocab(8)
```

- [ ] **Step 2: Run transform tests and verify they fail because decode and vocabulary validation do not exist.**

Run: `TL_BACKEND=torch pytest tests/transforms/test_graph_tokenizer.py -q`

Expected: failure on the new round-trip and permutation tests.

- [ ] **Step 3: Implement an unambiguous serialization grammar, inverse parser, canonical tie-breaking and token-id validation.**

Use node-start/node-end and component-separator tokens in the serialized grammar, store traversal metadata needed for edge reconstruction, and reject any encoder result outside the model embedding range.

- [ ] **Step 4: Run transform tests and verify they pass.**

Run: `TL_BACKEND=torch pytest tests/transforms/test_graph_tokenizer.py -q`

Expected: all serialization, BPE and tokenizer tests pass.

### Task 3: Torch-Only Official Encoder Adapter

**Files:**
- Create: `examples/graph_tokenizer/official_model_adapter.py`
- Create: `tests/models/test_graph_tokenizer_official_adapter.py`
- Modify: `examples/graph_tokenizer/graph_tokenizer_trainer.py`

**Interfaces:**
- Produces `require_paper_runtime()` which checks Torch, CUDA and Transformers lazily.
- Produces `create_paper_model(encoder_type, vocab_size, pad_token_id, task_type, output_dim, pooling)`.
- Produces `PaperGraphTokenizerModel.forward(input_ids, attention_mask, task='mlm'|'supervised')`.

- [ ] **Step 1: Write failing adapter tests using a local tiny `BertConfig` fixture.**

```python
model = create_paper_model("bert", vocab_size=16, pad_token_id=0,
                           task_type="regression", output_dim=1, pooling="mean")
assert model(input_ids, attention_mask, task="mlm").shape == (2, 4, 16)
assert model(input_ids, attention_mask, task="supervised").shape == (2, 1)
```

- [ ] **Step 2: Run adapter tests and verify they fail because the adapter module does not exist.**

Run: `TL_BACKEND=torch pytest tests/models/test_graph_tokenizer_official_adapter.py -q`

Expected: import failure for `official_model_adapter`.

- [ ] **Step 3: Implement delayed imports, HF BERT initialization, official GTE loading, pooling, MLM head and task head.**

`bert` creates `transformers.BertModel(BertConfig(...))`. `gte` calls `AutoModel.from_pretrained('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True)`, resizes graph-token embeddings, preserves pad ID zeroing, and reports the loaded revision in the manifest.

- [ ] **Step 4: Run adapter tests and verify they pass.**

Run: `TL_BACKEND=torch pytest tests/models/test_graph_tokenizer_official_adapter.py -q`

Expected: BERT fixture tests pass; GTE download tests are marked integration-only.

### Task 4: Paper Training, Metrics, Checkpoints And Presets

**Files:**
- Modify: `examples/graph_tokenizer/graph_tokenizer_trainer.py`
- Create: `examples/graph_tokenizer/configs/qm9_paper.json`
- Create: `examples/graph_tokenizer/configs/molhiv_paper.json`
- Create: `examples/graph_tokenizer/configs/peptides_struct_paper.json`
- Modify: `tests/models/test_graph_transformer.py`
- Create: `tests/models/test_graph_tokenizer_paper_protocol.py`

**Interfaces:**
- Produces CLI options `--protocol`, `--target-property`, `--paper-config`, `--device` and `--num-workers`.
- Produces `validate_paper_preflight(args)` and `run_paper_experiment(args)`.
- Produces `save_paper_checkpoint(...)` and `restore_paper_checkpoint(...)` containing model, optimizer, scheduler, tokenizer, epoch, best validation metric and Python/NumPy/Torch/CUDA RNG states.

- [ ] **Step 1: Write failing tests for invalid paper runtime, target-property enforcement, OGB evaluator parity, test-only-after-restore and checkpoint state restoration.**

```python
with pytest.raises(ValueError, match="target_property"):
    parse_args(["--protocol", "paper", "--dataset", "qm9"])
assert compute_paper_metric(molhiv_spec, y_true, y_pred) == ogb_value
assert history[-1]["test_evaluations"] == 1
```

- [ ] **Step 2: Run protocol tests and verify they fail because paper mode and checkpoint restoration do not exist.**

Run: `TL_BACKEND=torch pytest tests/models/test_graph_tokenizer_paper_protocol.py -q`

Expected: parser and runtime failures for absent paper-mode interfaces.

- [ ] **Step 3: Implement the paper protocol.**

Use `torch.optim.AdamW`, linear warmup schedulers, DataLoader shuffle on train only, paper seed sequence `42..46`, per-dataset presets, train-only tokenizer fitting, validation-driven early stopping, one final test evaluation after loading the best checkpoint, OGB `Evaluator('ogbg-molhiv')`, and average task-wise MAE for Peptides-struct. Configure QM9 with 16 named targets and training-split normalization.

- [ ] **Step 4: Run model and paper-protocol tests and verify they pass.**

Run: `TL_BACKEND=torch pytest tests/models/test_graph_transformer.py tests/models/test_graph_tokenizer_paper_protocol.py -q`

Expected: existing trainer tests and new protocol tests pass.

### Task 5: C++ Backend, Documentation And End-To-End Smoke Verification

**Files:**
- Modify: `third_party/graph_bpe_cpp/setup.py`
- Modify: `third_party/graph_bpe_cpp/__init__.py`
- Create: `tests/transforms/test_graph_bpe_cpp_install.py`
- Modify: `examples/graph_tokenizer/README.md`

**Interfaces:**
- Produces an installed extension importable as `third_party.graph_bpe_cpp._graph_bpe`.
- Produces remote commands for data preparation, one-epoch smoke run and five-run paper benchmark.

- [ ] **Step 1: Write a failing import test for the installed C++ extension.**

```python
module = importlib.import_module("third_party.graph_bpe_cpp._graph_bpe")
assert hasattr(module, "GraphBPE")
```

- [ ] **Step 2: Build the extension and run the import test to verify the current package-path mismatch.**

Run: `TL_BACKEND=torch python third_party/graph_bpe_cpp/setup.py build_ext --inplace`

Run: `TL_BACKEND=torch pytest tests/transforms/test_graph_bpe_cpp_install.py -q`

Expected: current import test fails before package-name correction.

- [ ] **Step 3: Align the extension module name with the runtime import and document required paper dependencies and commands.**

The README must distinguish GammaGL mode from paper mode, list the optional Torch/HuggingFace/OGB dependencies, require CUDA for paper mode, and show preflight, smoke and full five-run commands.

- [ ] **Step 4: Build the extension and run the complete GraphTokenizer test suite.**

Run: `TL_BACKEND=torch python third_party/graph_bpe_cpp/setup.py build_ext --inplace`

Run: `TL_BACKEND=torch pytest tests/datasets/test_molecule_benchmark_datasets.py tests/transforms/test_graph_tokenizer.py tests/transforms/test_graph_bpe_cpp_install.py tests/models/test_graph_transformer.py tests/models/test_graph_tokenizer_official_adapter.py tests/models/test_graph_tokenizer_paper_protocol.py -q`

Expected: all selected tests pass; GTE-network tests are explicit integration tests and are run on the remote server.

## Plan Self-Review

- Data correctness, reversible serialization, official encoders, paper training protocol, C++ packaging and remote documentation each map to an independently testable task.
- The public GammaGL import boundary is preserved because all Torch/HuggingFace imports remain inside experiment-only code paths.
- No task changes generic GammaGL graph or message-passing APIs, and no task creates a Git commit.
