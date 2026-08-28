# Native TensorLayerX GraphTokenizer Models Design

## Objective

Move the strict paper BERT-Small and GTE-Base implementations into
`gammagl/models/graph_bert.py` and `gammagl/models/graph_gte.py`, respectively.
The paper protocol must instantiate those public model classes directly and
must no longer load `examples/graph_tokenizer/official_model_adapter.py`.

Delete `official_model_adapter.py` after its required behavior and regression
coverage have been migrated. Model internals must use TensorLayerX public
layers and tensor operations. The implementation should remain portable across
TensorLayerX backends where those public APIs are supported, while the strict
paper protocol continues to validate and run with the TensorLayerX PyTorch
backend.

## Scope

This change covers model construction, forward interfaces, architecture
metadata, paper-protocol model loading, and model regression tests. It does not
change graph serialization, BPE, tokenizer training, official dataset splits,
loss functions, optimizers, schedules, early stopping, or result aggregation.

The already-running GTE processes use the adapter module loaded into their
process memory. They are not interrupted by this refactor, but their results
remain adapter-version results. New native-TLX results must be reported as a
different implementation and require new preflight and formal runs before they
can be compared with adapter-version results.

## Selected Architecture

Use two public model files and no additional model adapter or shared model
module:

- `graph_bert.py` owns the BERT-Small model and reusable TensorLayerX
  Transformer primitives that are genuinely common to both encoders.
- `graph_gte.py` owns GTE-specific rotary embeddings, rotary attention, gated
  MLP blocks, and the `GraphGTE` public model. It may import small common
  utilities from `graph_bert.py`, but it must not inherit an encoder that adds
  absolute position embeddings.
- `paper_protocol.py` loads and instantiates `GraphBERT` or `GraphGTE`
  directly. It retains the PyTorch-only paper training loop because the paper
  software specification and current optimizer/checkpoint implementation are
  PyTorch-specific.

This avoids duplicated generic utilities while ensuring the GTE computation
graph is not accidentally a long-window BERT.

## Public Model Contract

Both public classes retain the existing constructor inputs used by the legacy
trainer:

```python
GraphBERT(
    vocab_size,
    output_dim,
    hidden_size=512,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=2048,
    max_position_embeddings=768,
    dropout_rate=0.1,
    ...,
)

GraphGTE(
    vocab_size,
    output_dim,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=8192,
    dropout_rate=0.1,
    ...,
)
```

They add explicit paper-compatible inputs such as `pad_token_id`, `task_type`,
`pooling`, `layer_norm_eps`, and attention dropout without removing the
existing arguments.

The forward interface is backward compatible:

- `model(input_ids, attention_mask=mask)` returns the existing dictionary with
  `logits`, `mlm_logits`, `pooled_output`, and `last_hidden_state`.
- `model(input_ids, attention_mask=mask, task="mlm")` returns MLM logits with
  shape `[batch, sequence, vocabulary]`.
- `model(input_ids, attention_mask=mask, task="supervised")` returns task
  logits with shape `[batch, output_dim]`.
- Unsupported task names raise `ValueError`.

Both classes expose `manifest()` with encoder name, pooling method, vocabulary
size, hidden size, layer count, attention heads, FFN size, activation,
dropouts, position encoding, maximum sequence length, LayerNorm epsilon, and
trainable/total parameter counts.

## Native TensorLayerX BERT-Small

`GraphBERT` implements a standard post-normalization BERT encoder using only
TensorLayerX public APIs:

- trainable token embeddings;
- learned absolute position embeddings;
- token-type embeddings;
- embedding LayerNorm with epsilon `1e-12` and dropout `0.1`;
- four Transformer layers by default;
- explicit query, key, and value projections;
- scaled dot-product multi-head self-attention with padding-mask bias;
- attention probability dropout `0.1`;
- attention residual followed by LayerNorm;
- GELU feed-forward network `512 -> 2048 -> 512`;
- feed-forward residual followed by LayerNorm;
- learned absolute position limit of 768 tokens;
- truncated-normal weight initialization with standard deviation `0.02`, zero
  biases, unit LayerNorm scale, and zero LayerNorm bias.

The supervised head preserves the adapter behavior: mean or CLS pooling,
followed by `hidden_size -> hidden_size/2 -> output_dim` with ReLU and dropout.
The MLM projection maps hidden states to the dynamic graph-token vocabulary.

## Native TensorLayerX GTE-Base

`GraphGTE` is a separate native TensorLayerX encoder rather than a parameter
preset over the BERT encoder. Its default strict-paper architecture is:

- 12 layers, hidden size 768, 12 heads, head size 64;
- intermediate size 3072;
- token and single token-type embeddings without learned absolute position
  embeddings;
- embedding and block LayerNorm epsilon `1e-12`;
- hidden and attention dropout `0.1`, matching the current paper override;
- packed QKV projection followed by an output projection;
- RoPE applied to query and key tensors before attention scores;
- `rope_theta=20000` and official GTE NTK scaling factor `8.0`;
- post-normalization residual attention blocks;
- official GTE gated GELU MLP: a bias-free projection to twice the
  intermediate size, split into value and gate halves, GELU on the gate,
  elementwise gating, and projection back to hidden size;
- maximum sequence length 8192;
- truncated-normal initialization with standard deviation `0.02` and an
  explicitly zeroed padding embedding row.

The native NTK rotary cache follows the locally cached official GTE behavior:
for lengths beyond the base maximum it adjusts the rotary base and inverse
frequencies using the configured scaling factor. Cosine and sine tables are
computed with TensorLayerX `arange`, `einsum`, `concat`, `cos`, and `sin`, and
are applied with TensorLayerX reshape/split/concat operations.

The model is created in the TensorLayerX default floating-point dtype. The
strict paper runtime verifies FP32 trainable parameters before constructing
AdamW, preserving the previous protection against first-step half-precision
overflow without using PyTorch operations inside the model implementation.

## Masking, Pooling, and Padding

Attention masks use `1` for real tokens and `0` for padding. Each model turns
the mask into a broadcastable additive attention bias through TensorLayerX
operations. Padding positions cannot receive attention probability.

Mean pooling multiplies hidden states by the attention mask and divides by a
denominator clamped to at least one. CLS pooling selects sequence position
zero. An input longer than `max_position_embeddings` raises `ValueError`
before attention allocation.

The embedding at `pad_token_id` is initialized to zero. Padding invariance is
tested at the observable pooled/logit interface rather than by inspecting a
backend-specific parameter object.

## Paper Protocol Integration

`paper_protocol.py` removes `_load_adapter()` and replaces it with a focused
model loader that loads the two GammaGL model files. Runtime validation checks:

- `TL_BACKEND` is `torch` for strict paper runs;
- TensorLayerX is active with its PyTorch backend;
- CUDA is available when required;
- all model trainable parameters are FP32 before optimizer construction.

For each of the five seeds, the protocol instantiates `GraphBERT` or
`GraphGTE` with the dynamic tokenizer vocabulary, dataset output dimension,
padding token, configured pooling, and exact paper architecture. The remaining
MLM pretraining, downstream fine-tuning, early stopping, checkpointing, and
single final test evaluation remain unchanged.

No code path may import, dynamically load, or mention
`official_model_adapter.py` after migration.

## File Changes

- Modify `gammagl/models/graph_bert.py` with the native BERT implementation,
  shared TLX primitives, paper-compatible task interface, and manifest.
- Modify `gammagl/models/graph_gte.py` with native RoPE GTE blocks, gated MLP,
  task interface, and manifest.
- Modify `examples/graph_tokenizer/paper_protocol.py` to instantiate the two
  public model classes directly and perform paper runtime checks.
- Delete `examples/graph_tokenizer/official_model_adapter.py`.
- Modify `tests/models/test_graph_transformer.py` to own model architecture,
  forward, masking, RoPE, manifest, and gradient tests.
- Modify `tests/models/test_graph_tokenizer_paper_protocol.py` to verify direct
  model loading and removal of the adapter dependency.
- Delete `tests/models/test_graph_tokenizer_official_adapter.py` after its
  still-relevant assertions have been migrated.
- Update `examples/graph_tokenizer/README.md` only where it describes model
  construction or dependencies; training commands and paper configurations
  remain unchanged.

## Test-Driven Migration

Production changes follow red-green-refactor cycles:

1. Add a failing protocol test requiring direct model-file loading and no
   adapter path.
2. Add failing BERT tests for exact manifest values, output contracts,
   LayerNorm epsilon, absolute positions, padding behavior, and gradients.
3. Implement the minimum native BERT behavior needed to pass those tests.
4. Add failing GTE tests for the absence of absolute position embeddings,
   RoPE numerical behavior, NTK configuration, gated MLP, exact manifest,
   output contracts, and gradients.
5. Implement the minimum native GTE behavior needed to pass those tests.
6. Switch the paper protocol to the public classes, migrate the useful adapter
   tests, and delete adapter code and adapter-only tests.
7. Run the focused model and paper-protocol suites, then the complete
   GraphTokenizer-related suite.
8. Run BERT and GTE preflight commands and one tiny paper-protocol training
   cycle for each architecture in the pinned PyTorch environment.

RoPE tests compare a small deterministic TensorLayerX result against an
independently calculated reference rather than asserting only tensor shapes.
Gradient tests require finite, nonzero gradients through attention, MLM, and
the supervised head. Source-dependency tests fail if the adapter filename or
Hugging Face model constructors reappear in the strict native model path.

## Acceptance Criteria

The migration is complete only when all of the following are true:

- `official_model_adapter.py` and its adapter-only test file no longer exist.
- Repository search finds no runtime or test reference to the adapter.
- Strict paper mode instantiates `GraphBERT` and `GraphGTE` from their GammaGL
  model files.
- Model implementation files use TensorLayerX public layers and tensor
  operations and contain no direct PyTorch or Transformers import.
- BERT reports and executes the exact BERT-Small architecture.
- GTE reports and executes RoPE, NTK scaling, gated GELU MLP, and the exact
  12-layer GTE-Base dimensions.
- Legacy dictionary outputs and paper task-specific outputs both work.
- Focused and full GraphTokenizer regression suites pass.
- BERT and GTE preflights pass in the pinned environment.
- Tiny native-TLX BERT and GTE paper-protocol runs complete without traceback,
  CUDA OOM, NaN, or non-finite gradients and write valid checkpoints/results.
- Existing adapter-version formal results remain labeled as such; native-TLX
  formal results are generated separately.
