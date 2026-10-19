GraphTokenizer
==============

The native TensorLayerX implementation follows Zeyuan Guo, Enmao Diao,
Cheng Yang and Chuan Shi, `"Graph Tokenization for Bridging Graphs and
Transformers" <https://arxiv.org/abs/2603.11099>`_ (ICLR 2026), and the
`official implementation at revision 98343a6
<https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/tree/98343a6b025a48fbb6859cd812a12b81ec3ac3cc>`_.
Its actual serializer and loader behavior determines this API's input contract.
The paper's general reversibility description does not imply that the default
official token stream here can reconstruct graph identities.

API reference
-------------

The API catalogs link to the complete constructor and method contracts:

* :doc:`Models </api/gammagl.layers>`: :class:`~gammagl.models.GraphBERT`,
  :class:`~gammagl.models.GraphTokenTransformer` and :class:`~gammagl.models.GraphGTE`.
* :doc:`Datasets </api/gammagl.datasets>`: :class:`~gammagl.datasets.Mutagenicity`,
  :class:`~gammagl.datasets.GraphTokenizerDBLP` and :class:`~gammagl.datasets.AQSOL`.
* :doc:`Transforms </api/gammagl.transforms>`: serializers, :class:`~gammagl.transforms.GraphBPE`,
  :class:`~gammagl.transforms.GraphTokenizer` and their result containers.

Inputs and protocol
-------------------

:class:`~gammagl.transforms.GraphTokenizer` combines a serializer, integer BPE
and a frozen model vocabulary. It accepts simple graphs with scalar integer
node and edge labels; see
:class:`~gammagl.transforms.FrequencyGuidedEulerianSerializer` for shapes,
defaults and validation. The official protocol supports disconnected graphs
and isolated nodes, rejects empty graphs, loops, parallel edges and explicitly
directed metadata, and preserves the supplied COO arcs. Keep the release's
bidirectional arcs: one-way storage need not produce the same sequence.

The default ``official`` stream has no extra node references. It interleaves
the actual loader's node / retained edge tokens and omits the designated common
edge type. TU bulk and singleton getters differ; the serializer's ``dataset``
argument applies those rules. AQSOL adapters supply atomic and bond tokens.
Raw token IDs, BPE IDs and final model IDs are distinct alphabets. In particular,
raw IDs 0 through 7 are ordinary BPE inputs, and the pinned official vocabulary
always aliases ordinary raw token 0 to PAD. Unseen other labels map to UNK,
without being mistaken for learned BPE IDs.

The trainer's six presets explicitly select the following construction path:

.. list-table::
   :header-rows: 1

   * - Dataset / encoder
     - Serializer
     - BPE encoding
   * - Mutagenicity / BERT
     - feuler
     - all
   * - Mutagenicity / GTE
     - feuler
     - none (raw)
   * - DBLP_v1 / BERT
     - feuler
     - all
   * - DBLP_v1 / GTE
     - eulerian
     - none (raw)
   * - AQSOL / BERT
     - feuler
     - all
   * - AQSOL / GTE
     - feuler
     - all

The two TU GTE configurations override the default YAML; the other four inherit
its serializer / BPE mode. The trainer explicitly enables multiple starts.
``none`` still fits BPE rules and a covering vocabulary, then skips rule
application before model input. All six presets default to ``--fit-scope all``:
frequencies, BPE and vocabulary use train, val, then test graphs in the pinned
official loader's concatenation order. This is transductive preprocessing;
regression normalization and model optimization still use training labels only.
The explicit ``--fit-scope train`` inductive variant writes under ``train_only/``
and cannot be aggregated with all-scope runs. Generic ``GraphTokenizer.fit``
defaults to ``fit_scope="train"``; its caller supplies the corresponding corpus.
These protocol checks do not establish full training or benchmark equivalence.

The separate, opt-in ``protocol="local_refs"`` grammar supports graph recovery.
Only it normalizes single / symmetric undirected COO equivalently and emits
first-visit node references. Neither protocol guarantees global isomorphism
canonicalization. :meth:`~gammagl.transforms.GraphTokenizer.decode_graph` and
the serializer's recovery methods raise :class:`NotImplementedError` for
``official``, even with full result metadata. :meth:`~gammagl.transforms.GraphBPE.decode`
only expands compressed tokens; it cannot recover omitted graph information.

Minimal offline example
-----------------------

The models consume final integer IDs of shape ``(B, L)``, not ``edge_index``.
This example uses already encoded scalar tokens and the default generic loader
rules. For benchmark graphs, supply the corresponding serializer ``dataset``.

.. code-block:: python

   import tensorlayerx as tlx
   from gammagl.models import GraphBERT
   from gammagl.transforms import GraphBPE, GraphTokenizer

   graph = dict(edge_index=[[0, 1], [1, 0]], x=[11, 13],
                edge_attr=[0, 0], num_nodes=2)
   tokenizer = GraphTokenizer(bpe=GraphBPE(num_merges=2)).fit([graph])
   result = tokenizer.encode_graph(graph)
   ids, mask = tokenizer.pad_token_sequences([result.input_ids], max_length=16)
   model = GraphBERT(tokenizer.max_token_id + 1, output_dim=2,
                     hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=4, intermediate_size=32,
                     max_position_embeddings=16)
   model.set_eval()
   logits = model(tlx.convert_to_tensor(ids, dtype=tlx.int64),
                  tlx.convert_to_tensor(mask, dtype=tlx.int64), task="supervised")
   assert tuple(logits.shape) == (1, 2)

Task modes, backends and evaluation
-----------------------------------

:class:`~gammagl.models.GraphBERT` inherits
:class:`~gammagl.models.GraphTokenTransformer`'s parameters and task modes.
:class:`~gammagl.models.GraphGTE` implements the same forward contract with RoPE
and gated GELU. ``task="mlm"`` returns ``(B, L, vocab_size)`` logits,
``"supervised"`` returns raw ``(B, output_dim)`` predictions, ``"pooled"``
returns ``(B, hidden_size)`` features, and ``None`` returns all heads plus hidden
states. ``task_type`` is metadata, not a softmax or loss selector. Pass an
attention mask explicitly to hide padding. Excessive sequence lengths raise
:class:`ValueError`; no model mode groups serializations automatically.

The supported training paths are TLX torch and TensorFlow eager. Other TLX
backends are not validated for this pipeline. Device selection occurs before
seeding/data operations (and before TLX import for the direct TensorFlow CLI).
``cpu`` hides TensorFlow GPUs; ``cuda:N`` selects the Nth OS-visible GPU and
uses logical GPU 0 after selection, with memory growth enabled. Invalid or
unavailable devices and incompatible initialized runtimes raise an error.

The native trainer exports a per-seed ``run_manifest.json`` before training,
with resolved arguments, source hashes/version, backend/runtime and the actual
model constructor configuration. Git checkouts record the commit and dirty
status; source snapshots record relevant file hashes without inferring a
commit. Full logs and the complete dirty diff must still be archived separately.
A relative, content-addressed preprocessing
bundle stores the fitted tokenizer (including BPE/vocabulary), train-only label
normalizer and input-file/split summaries. The independent artifact version is
1; tokenizer schema 7 and cache 9 are unchanged. Checkpoints bind the bundle
fingerprint; successful summaries also record manifest/checkpoint hashes.

Copy the complete dataset/encoder output directory, then use
``examples.graph_tokenizer.artifacts.load_run(run_directory, device="cpu")``
in a new process with the same TLX backend. It returns the evaluation-mode
model, tokenizer, normalizer and run manifest, without raw data, preprocessing
caches or Hugging Face access. Loading restores full downstream weights and
does not refit the tokenizer or fetch encoder initialization. Corrupted/mixed
assets are rejected. Legacy runs without export manifests require their
original paired assets through the low-level helpers. Pickles/checkpoints must
be trusted; file hashes detect mismatches, not malicious provenance.

GTE constructors initialize
random weights. :meth:`~gammagl.models.GraphGTE.from_pretrained` loads the
`pinned GTE encoder
<https://huggingface.co/Alibaba-NLP/gte-multilingual-base/tree/9bbca17d9273fd0d03d5725c7a4b0f6b45142062>`_,
checks its configuration, checkpoint SHA-256, tensor shapes and complete
coverage including resized word embeddings. Existing rows are retained and
new rows use N(0, 0.02). The fitted model-to-official vocabulary permutation
aligns embedding and MLM weight / bias rows together; PAD is zero and has no
lookup gradient. MLM and graph task heads are newly initialized.
The API signature's random-model defaults differ from some pretrained values.
The pinned default sets ``encoder.reset_weights=true``. The AQSOL / GTE preset
therefore uses :meth:`~gammagl.models.GraphGTE.from_official_config` to construct
the official architecture with random normal weights; it does not load a
checkpoint. The two TU / GTE presets load pretrained weights. Initialization
summaries distinguish ``official_random`` / ``official_pretrained`` and record
``initialization_matches_preset``. They keep ``reproduction=false``: initialization
checks alone do not establish a reproduced experiment. Explicit development
overrides use ``random_development`` instead.

The example trainer validates single-label classification by softmaxing each
serialization's logits, averaging probabilities per ``graph_id``, then taking
argmax. Validation selection and final testing use the same rule across batch
boundaries and unequal variant counts. For example, ``[[10, 0], [0, 2], [0, 2]]``
predicts class 1. Regression averages predictions, inverse-normalizes and
computes raw-scale MAE. Non-finite logits, losses, gradients, updated weights
or metrics raise :class:`FloatingPointError`; failed runs are not saved as
successful summaries. These are trainer checks, not guarantees of model forward.

Fine-tuning enables R-Drop with weight 1.0 for the Mutagenicity / GTE and
DBLP / GTE presets; the other four presets disable it. Classification averages
two stochastic cross-entropies and adds the weighted symmetric KL, retaining
both gradient paths. Regression uses MSE between predictions when explicitly
enabled. Accumulation normalizes by the actual targets once. Override with
``--finetune-use-consistency-regularization``,
``--no-finetune-use-consistency-regularization`` and
``--finetune-consistency-alpha`` (finite and non-negative). The fine-tuning
summary records the effective switch and weight; evaluation remains unchanged.

Artifacts and data preparation
------------------------------

Tokenizer schema is 7, serializer protocol is 5; preprocessing cache format is 9.
Rebuild obsolete tokenizers and caches. Formal checkpoints bind the actual
tokenizer's :meth:`~gammagl.transforms.GraphTokenizer.fingerprint`, including
protocol, serializer, rules, vocabulary, fit scope and fitted statistics. Loading checks
schema, explicit fit scope and fingerprint before assigning weights. Equal embedding shapes are
insufficient; standalone BPE rule loading is not tokenizer migration. Bare
token lists have no schema provenance.

The three benchmark adapters preserve fixed train / val / test indices.
They require trusted pickle data and scalar features; see
:class:`~gammagl.datasets.Mutagenicity`, :class:`~gammagl.datasets.GraphTokenizerDBLP`
and :class:`~gammagl.datasets.AQSOL` for the shared constructor and item contract.
Constructing a dataset does not download the shared release bundle.

Use the repository's executable
`installation, preparation and smoke commands
<https://github.com/baihchou8787/GammaGL/blob/pr/graphtokenizer-paper-8-29-2/examples/graph_tokenizer/README.md>`_.
After installing the paper extra, preparation from a repository checkout is:

.. code-block:: bash

   python examples/graph_tokenizer/graph_tokenizer_trainer.py --prepare-data --data-root data

Without real data or downloads, check the training pipeline using:

.. code-block:: bash

   TL_BACKEND=torch python examples/graph_tokenizer/graph_tokenizer_trainer.py \
     --smoke --dataset aqsol --encoder bert --seeds 42 --output-dir /tmp/graph-tokenizer-smoke

For a development GTE smoke, select ``--encoder gte --allow-random-gte-init``;
this is not a pretrained GTE reproduction. Inputs above the tokenizer limit
are rejected rather than truncated; augmented overlength inputs fall back to
the original encoded input. Local offline fixtures validate protocol behavior,
not the six previously reported benchmark records. Those results and their
attribution remain unchanged.
