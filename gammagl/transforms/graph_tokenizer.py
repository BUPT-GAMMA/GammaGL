from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Dict, List

from .base_transform import BaseTransform
from .graph_bpe import GraphBPE
from .graph_serializer import (FrequencyGuidedEulerianSerializer,
                               GraphSerializationResult, SERIALIZER_PROTOCOL_VERSION)


TOKENIZER_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class GraphTokenizerSpecialTokens:
    r"""Model vocabulary special IDs (not the raw serializer / BPE alphabet).

    Parameters
    ----------
    pad_token_id: int
        Padding ID. (default: ``0``)
    unk_token_id: int
        Unknown model token ID. (default: ``1``)
    mask_token_id: int
        MLM replacement ID. (default: ``2``)
    cls_token_id: int
        Sequence start ID. (default: ``3``)
    sep_token_id: int
        Sequence end ID. (default: ``4``)
    node_start_token_id: int
        Reserved node-start ID; not inserted by the default serializer.
        (default: ``5``)
    node_end_token_id: int
        Reserved node-end ID; not inserted by the default serializer.
        (default: ``6``)
    component_sep_token_id: int
        Reserved model ID; official raw component token 7 passes through
        BPE / vocabulary mapping as an ordinary token. (default: ``7``)

    Notes
    -----
    This dataclass is immutable. The GTE trainer swaps PAD / UNK to 1 / 0.
    official preserves the pinned VocabManager's ordinary raw 0 / PAD alias;
    it is not an additional node-reference or graph-recovery mechanism.
    """
    pad_token_id: int = 0
    unk_token_id: int = 1
    mask_token_id: int = 2
    cls_token_id: int = 3
    sep_token_id: int = 4
    node_start_token_id: int = 5
    node_end_token_id: int = 6
    component_sep_token_id: int = 7


@dataclass
class GraphTokenizationResult:
    r"""One unpadded model input and its serialization provenance.

    Parameters
    ----------
    input_ids: list[int]
        Model vocabulary IDs of length L, including CLS / SEP if enabled.
    attention_mask: list[int]
        L ones; padding is added by separate batch helpers.
    serialized_token_ids: list[int]
        Length S before BPE and model vocabulary mapping. official keeps
        the raw loader stream; local_refs stores its normalized stream.
        Retains unseen raw labels even when input_ids maps them to UNK.
    metadata: dict
        Serializer metadata, BPE counts, required model vocabulary size,
        special-token flag, tokenizer schema and fit graph-ID hash.
        Retaining this metadata does not make official streams reversible.
    """

    input_ids: List[int]
    attention_mask: List[int]
    serialized_token_ids: List[int]
    metadata: Dict[str, Any]


@dataclass
class GraphTokenizerMLMBatch:
    r"""Padded Python-list batch returned by GraphTokenizer's MLM helpers.

    Parameters
    ----------
    input_ids: list[list[int]]
        Masked model IDs of shape (B, max_length).
    attention_mask: list[list[int]]
        Shape (B, max_length); 1 for original tokens, 0 for padding.
    labels: list[list[int]]
        Shape (B, max_length); original token IDs at prediction targets and
        -100 elsewhere (ignored by the trainer loss).
    metadata: dict
        mask_prob, max_length, num_graphs and num_mlm_labels. The batch is
        not a TLX tensor; convert lists to backend tensors before modeling.
    """

    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    labels: List[List[int]]
    metadata: Dict[str, Any]


class GraphTokenizer(BaseTransform):
    r"""Compose graph serialization, BPE and a fitted model vocabulary.

    Based on `"Graph Tokenization for Bridging Graphs and Transformers"
    <https://arxiv.org/abs/2603.11099>`_ and the `pinned official code
    <https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/tree/98343a6b025a48fbb6859cd812a12b81ec3ac3cc>`_.
    The default official protocol has no node-reference tokens and does not
    support graph recovery. See :doc:`/notes/graph_tokenizer` for presets,
    evaluation aggregation and artifact compatibility.

    Parameters
    ----------
    serializer: FrequencyGuidedEulerianSerializer
        None creates the official Feuler serializer with dataset=None. Use
        its dataset argument to apply a benchmark's actual loader rules, or
        explicitly select protocol="local_refs" for GammaGL graph recovery.
        See the serializer for graph domains and input shapes.
        (default: :obj:`None`)
    bpe: GraphBPE
        None creates GraphBPE with 2000 merges, minimum frequency 2, Python
        backend. official sets merge_id_offset=1; local_refs protects model
        special IDs in its normalized stream. (default: :obj:`None`)
    special_tokens: GraphTokenizerSpecialTokens
        None uses the standard eight model IDs. (default: :obj:`None`)
    add_special_tokens: bool
        Add CLS / SEP after vocabulary encoding. (default: :obj:`True`)
    bpe_encode_mode: str
        "all" applies learned rules; "none" sends base tokens through the
        vocabulary unchanged. Both modes fit BPE and cover its rule IDs in
        the model vocabulary. (default: ``"all"``)

    Raises
    ------
    ValueError
        If the encoding mode, fit corpus / realization count, graph inputs,
        model vocabulary size, sequence limit or result versions are invalid.
    RuntimeError
        If encoding / decoding is attempted before fit, or a tokenizer has
        an obsolete schema or modified serializer configuration.
    NotImplementedError
        If decode_graph is requested for official protocol.
    ImportError
        If the selected BPE backend requires an unavailable native extension.

    Notes
    -----
    Each encode_graph returns a :class:`GraphTokenizationResult` with
    input_ids and attention_mask of length L, plus S serialization tokens.
    Batch encoding is ragged; pad_token_sequences creates (B, max_length)
    lists without truncation. Calling this transform mutates the input
    graph / dict with these fields and returns the same object.

    Fit only on training graphs. New labels map to UNK without learning
    evaluation data; ordinary raw token 0 always aliases PAD in official
    mode. A bare BPE codebook or equal model embedding shape does not prove
    tokenizer compatibility. Current schema / serializer protocol is 5;
    formal trainer caches use format 6 and checkpoints bind fingerprint().

    Examples
    --------
    >>> from gammagl.transforms import GraphTokenizer
    >>> graph = dict(edge_index=[[0, 1], [1, 0]], x=[11, 13],
    ...              edge_attr=[0, 0], num_nodes=2)
    >>> tokenizer = GraphTokenizer().fit([graph])
    >>> result = tokenizer.encode_graph(graph)
    >>> result.input_ids
    [3, 8, 9, 8, 4]
    >>> result.serialized_token_ids
    [11, 13, 11]
    """

    def __init__(
        self,
        serializer: FrequencyGuidedEulerianSerializer | None = None,
        bpe: GraphBPE | None = None,
        special_tokens: GraphTokenizerSpecialTokens | None = None,
        add_special_tokens: bool = True,
        bpe_encode_mode: str = "all",
    ):
        self.serializer = serializer or FrequencyGuidedEulerianSerializer()
        self.bpe = bpe or GraphBPE()
        if bpe_encode_mode not in ("all", "none"):
            raise ValueError("bpe_encode_mode must be all or none.")
        self.bpe_encode_mode = bpe_encode_mode
        if self.serializer.protocol == "official":
            # The pinned minBPE reserves max(base)+1 as a virtual separator.
            self.bpe.merge_id_offset = 1
        self.special_tokens = special_tokens or GraphTokenizerSpecialTokens()
        self._protect_special_tokens()
        self.add_special_tokens = bool(add_special_tokens)
        self._fitted = False
        self.vocabulary = {}
        self.id_to_vocabulary_token = {}
        self.original_token_ids = None
        self.fit_graph_ids_hash = None
        self.fit_num_realizations = 1
        self.schema_version = TOKENIZER_SCHEMA_VERSION

    def fit(self, graphs, graph_ids=None, num_realizations: int = 1):
        r"""Fit serializer statistics, BPE and vocabulary on graphs, returning self.

        Parameters
        ----------
        graphs: ``iterable``
            Nonempty training graph corpus in the serializer's supported domain.
        graph_ids: ``iterable``
            One provenance ID per graph; stored as a hash, not as input tokens.
            (default: :obj:`None`)
        num_realizations: int
            Positive number of requested starts per graph, capped at its node
            count; see realization_start_nodes. (default: ``1``)

        Raises
        ------
        ValueError
            If the corpus is empty, counts / ID alignment are invalid, or graph
            validation fails. Refit rebuilds the vocabulary and invalidates old
            tokenizer-bound downstream checkpoints.
        """
        self.schema_version = TOKENIZER_SCHEMA_VERSION
        graphs = list(graphs)
        if not graphs:
            raise ValueError("Cannot fit GraphTokenizer on an empty training corpus.")
        num_realizations = int(num_realizations)
        if num_realizations <= 0:
            raise ValueError("num_realizations must be positive.")
        self.fit_num_realizations = num_realizations
        self.fit_graph_ids_hash = None
        if graph_ids is not None:
            graph_ids = list(graph_ids)
            if len(graph_ids) != len(graphs):
                raise ValueError("graph_ids must align with the training graphs.")
            self.fit_graph_ids_hash = hashlib.sha256(
                repr(tuple(graph_ids)).encode("utf-8")).hexdigest()
        self._protect_special_tokens()
        self.serializer.fit(graphs)
        serialized_sequences = []
        for graph in graphs:
            for start_node in self.realization_start_nodes(
                    graph, num_realizations):
                serialized_sequences.append(self._normalize_serialized_tokens(
                    self.serializer.serialize(
                        graph, start_node=start_node).token_ids))
        self.original_token_ids = {
            int(token) for sequence in serialized_sequences for token in sequence}
        self.bpe.fit(serialized_sequences)
        # Match the official protocol: BPE IDs are an intermediate alphabet,
        # then a frozen train-built contiguous vocabulary maps them to model IDs.
        # This permits inductive val/test encoding through UNK without changing
        # BPE rules or admitting evaluation graphs into vocabulary fitting.
        base_ids = {int(token) for sequence in serialized_sequences for token in sequence}
        merge_ids = {int(rule[2]) for rule in self.bpe.codebook.merge_rules}
        special_ids = set() if self.serializer.protocol == "official" else self._special_token_ids()
        token_ids = sorted((base_ids | merge_ids) - special_ids)
        if self.serializer.protocol == "official" and 0 in token_ids:
            # Official VocabManager reserves -pad_id=0, aliasing ordinary 0.
            token_ids.remove(0)
        start = self._reserved_token_count()
        self.vocabulary = {token: start + index for index, token in enumerate(token_ids)}
        if self.serializer.protocol == "official":
            self.vocabulary[0] = self.special_tokens.pad_token_id
        self.id_to_vocabulary_token = {
            vocab_id: token for token, vocab_id in self.vocabulary.items()}
        self._fitted = True
        self._fit_configuration = self._configuration()
        return self

    def _configuration(self):
        return dict(serializer=self.serializer.configuration(),
                    bpe_encode_mode=self.bpe_encode_mode,
                    special_tokens=vars(self.special_tokens), add_special_tokens=self.add_special_tokens,
                    merge_id_offset=self.bpe.merge_id_offset,
                    protected_token_ids=sorted(self.bpe.protected_token_ids))

    def fingerprint(self):
        r"""Return a SHA-256 string binding configuration and fitted token state.

        Covers serializer / protocol, source revision, fitted frequencies, BPE
        rules, original IDs, model vocabulary, special IDs and fit provenance.
        Raises RuntimeError for unfitted / incompatible tokenizer state. It is
        used for artifact matching, not as an authentication mechanism.
        """
        self._require_fitted()
        payload = dict(configuration=self._configuration(), schema=self.schema_version,
                       frequencies=sorted(self.serializer.frequency_map.items()),
                       vocabulary=sorted(self.vocabulary.items()),
                       merge_rules=self.bpe.codebook.merge_rules,
                       original_token_ids=sorted(self.original_token_ids),
                       fit_graph_ids_hash=self.fit_graph_ids_hash,
                       fit_num_realizations=self.fit_num_realizations)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def encode_tokens(self, token_ids: List[int]) -> List[int]:
        r"""Encode a pre-BPE serialized token list, returning model IDs of length L.

        token_ids must be in the same alphabet as encode_graph's
        serialized_token_ids (raw official, normalized local_refs). Uses the
        fitted rules / OOV mapping and adds CLS / SEP if configured. Raises
        RuntimeError if the tokenizer is unfitted or incompatible.
        """
        encoded = self._encode_token_sequences([token_ids])[0]
        if not self.add_special_tokens:
            return encoded
        return [self.special_tokens.cls_token_id, *encoded, self.special_tokens.sep_token_id]

    @property
    def max_token_id(self) -> int:
        r"""Largest currently assigned model token ID, including reserved IDs.

        Model vocab_size must be at least max_token_id + 1 after fitting.
        """
        return max(
            max(self.vocabulary.values(), default=self._reserved_token_count() - 1),
            self.special_tokens.pad_token_id,
            self.special_tokens.unk_token_id,
            self.special_tokens.mask_token_id,
            self.special_tokens.cls_token_id,
            self.special_tokens.sep_token_id,
            self.special_tokens.node_start_token_id,
            self.special_tokens.node_end_token_id,
            self.special_tokens.component_sep_token_id,
        )

    def validate_model_vocab(self, vocab_size: int) -> None:
        r"""Raise ValueError if vocab_size cannot represent the current model IDs.

        vocab_size is an integer embedding / output vocabulary capacity. Returns
        None on success; call after fit for the final vocabulary requirement.
        """
        required_vocab_size = self.max_token_id + 1
        if int(vocab_size) < required_vocab_size:
            raise ValueError(
                f"Model vocab_size={vocab_size} is smaller than the tokenizer "
                f"requirement {required_vocab_size}.")

    def encode_graph(
            self, graph: Any, start_node: int | None = None
    ) -> GraphTokenizationResult:
        r"""Encode one graph as a GraphTokenizationResult without padding.

        graph and optional start_node follow the serializer's contract; None
        selects its default start. Returns full serialization provenance and
        final model IDs. Raises RuntimeError for unfitted / incompatible state
        and ValueError for unsupported graph inputs or invalid starts.
        """
        self._require_fitted()
        serialized = self.serializer.serialize(graph, start_node=start_node)
        serialized_token_ids = self._normalize_serialized_tokens(serialized.token_ids)
        input_ids = self.encode_tokens(serialized_token_ids)
        return GraphTokenizationResult(
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            serialized_token_ids=serialized_token_ids,
            metadata={
                "serializer": serialized.metadata,
                "bpe": {
                    "num_merge_rules": len(self.bpe.codebook.merge_rules),
                    "vocab_size": self.bpe.codebook.vocab_size,
                },
                "required_vocab_size": self.max_token_id + 1,
                "add_special_tokens": self.add_special_tokens,
                "tokenizer_schema_version": self.schema_version,
                "fit_graph_ids_hash": self.fit_graph_ids_hash,
            },
        )

    def realization_start_nodes(self, graph: Any, num_realizations: int):
        r"""Return at most num_realizations start nodes for graph.

        official uses evenly spaced original IDs starting at 0. local_refs uses
        first-visit order (or [None] for a single realization). Returns a list of
        node IDs / None. Nonpositive counts and empty official graphs raise
        ValueError. Disconnected official serialization resets component starts.
        """
        num_realizations = int(num_realizations)
        if num_realizations <= 0:
            raise ValueError("num_realizations must be positive.")
        if self.serializer.protocol == "official":
            total_nodes = self.serializer._infer_num_nodes(
                graph, self.serializer._edge_pairs(self.serializer._get_graph_attr(graph, "edge_index")))
            if total_nodes <= 0:
                raise ValueError("Graph has no nodes, cannot serialize.")
            samples = min(num_realizations, total_nodes)
            step = max(1, total_nodes // samples)
            return [(index * step) % total_nodes for index in range(samples)]
        if num_realizations == 1:
            return [None]
        order = self.serializer.serialize(graph).metadata.get("local_to_original")
        if order is None:
            value = (graph.get("num_nodes") if isinstance(graph, dict)
                     else getattr(graph, "num_nodes", None))
            if callable(value):
                value = value()
            order = list(range(int(value)))
        total_nodes = len(order)
        if total_nodes <= 0:
            return [None]
        actual_samples = min(num_realizations, total_nodes)
        step = max(1, total_nodes // actual_samples)
        return [order[(index * step) % total_nodes] for index in range(actual_samples)]

    def decode_graph(self, encoding) -> Dict[str, Any]:
        r"""Recover a graph only with an explicitly selected local_refs serializer.

        Parameters
        ----------
        encoding: GraphTokenizationResult or list[int]
            A complete result restores original node IDs from provenance and
            retains raw labels even if the model saw UNK. Bare model IDs can
            only restore first-visit node IDs and cannot invert OOV information.

        Returns
        -------
        dict
            edge_index (2, M), x (N,), edge_attr (M,), num_nodes, as Python
            lists / integers. M counts undirected canonical edges.

        Raises
        ------
        NotImplementedError
            For official protocol, even when a full result is supplied.
        RuntimeError
            For an unfitted or incompatible tokenizer.
        ValueError
            For incompatible result versions, missing special tokens or invalid
            local_refs grammar / provenance. Bare lists have no version metadata.
        """
        self._require_fitted()
        if isinstance(encoding, GraphTokenizationResult):
            serializer_metadata = encoding.metadata.get("serializer", {})
            if (encoding.metadata.get("tokenizer_schema_version") != TOKENIZER_SCHEMA_VERSION
                    or serializer_metadata.get("protocol_version") != SERIALIZER_PROTOCOL_VERSION):
                raise ValueError("Incompatible tokenizer or serializer protocol version.")
            input_ids = encoding.input_ids
        else:
            input_ids = list(encoding)
        if self.serializer.protocol == "official":
            raise NotImplementedError(
                "Official tokens omit node identity; cannot reconstruct a graph. "
                "Use the explicit local_refs protocol for graph round trips.")
        if self.add_special_tokens:
            if (
                    len(input_ids) < 2
                    or input_ids[0] != self.special_tokens.cls_token_id
                    or input_ids[-1] != self.special_tokens.sep_token_id):
                raise ValueError("Encoded graph must begin with CLS and end with SEP tokens.")
            input_ids = input_ids[1:-1]
        if isinstance(encoding, GraphTokenizationResult):
            # The full result retains the lossless stream even when a held-out
            # token was mapped to UNK for the model input.
            token_ids = self._denormalize_serialized_tokens(
                encoding.serialized_token_ids)
            return self.serializer.restore_input_metadata(
                GraphSerializationResult(token_ids, serializer_metadata))
        serialized = self.bpe.decode([
            self.id_to_vocabulary_token.get(token, token)
            for token in input_ids
        ])
        token_ids = self._denormalize_serialized_tokens(serialized)
        return self.serializer.deserialize(token_ids)

    def batch_encode_graphs(self, graphs) -> List[GraphTokenizationResult]:
        r"""Encode graphs into a ragged list of GraphTokenizationResult objects.

        Uses each serializer's default start, without padding or per-graph
        aggregation. Same graph / fitting exceptions as encode_graph apply.
        """
        self._require_fitted()
        graphs = list(graphs)
        serialized_results = [
            self.serializer.serialize(graph) for graph in graphs
        ]
        serialized_token_ids = [
            self._normalize_serialized_tokens(result.token_ids)
            for result in serialized_results
        ]
        encoded_sequences = self._encode_token_sequences(serialized_token_ids)
        results = []
        for serialized, normalized, encoded in zip(
                serialized_results, serialized_token_ids, encoded_sequences):
            encoded = list(map(int, encoded))
            if self.add_special_tokens:
                encoded = [
                    self.special_tokens.cls_token_id,
                    *encoded,
                    self.special_tokens.sep_token_id,
                ]
            results.append(GraphTokenizationResult(
                input_ids=encoded,
                attention_mask=[1] * len(encoded),
                serialized_token_ids=normalized,
                metadata={
                    "serializer": serialized.metadata,
                    "bpe": {
                        "num_merge_rules": len(self.bpe.codebook.merge_rules),
                        "vocab_size": self.bpe.codebook.vocab_size,
                    },
                    "required_vocab_size": self.max_token_id + 1,
                    "add_special_tokens": self.add_special_tokens,
                    "tokenizer_schema_version": self.schema_version,
                    "fit_graph_ids_hash": self.fit_graph_ids_hash,
                },
            ))
        return results

    def _normalize_serialized_tokens(self, token_ids: List[int]) -> List[int]:
        if self.serializer.protocol == "official":
            return list(map(int, token_ids))
        reserved_token_count = self._reserved_token_count()
        component_sep_token_id = getattr(
            self.serializer,
            "component_sep_token_id",
            self.special_tokens.component_sep_token_id,
        )
        normalized = []
        for token in token_ids:
            token = int(token)
            if token == component_sep_token_id:
                normalized.append(self.special_tokens.component_sep_token_id)
            else:
                normalized.append(token + reserved_token_count)
        return normalized

    def _encode_token_sequences(self, token_sequences) -> List[List[int]]:
        self._require_fitted()
        prepared_sequences = [
            self._prepare_bpe_tokens(token_ids) for token_ids in token_sequences]
        encoded = (prepared_sequences if self.bpe_encode_mode == "none"
                   else self.bpe.batch_encode(prepared_sequences))
        return [
            self._vocabulary_encode(token_ids)
            for token_ids in encoded
        ]

    def _prepare_bpe_tokens(self, token_ids: List[int]) -> List[int]:
        if self.serializer.protocol == "official":
            # Training merge IDs share the numeric alphabet with raw labels.
            # An unseen raw label must not impersonate a learned merge token.
            # Official raw IDs are non-negative; -1 never has a vocabulary entry.
            return [int(token) if int(token) == 0 or int(token) in self.original_token_ids else -1
                    for token in token_ids]
        special_ids = self._special_token_ids()
        return [
            token if token in self.original_token_ids or token in special_ids
            else self.special_tokens.unk_token_id
            for token in map(int, token_ids)
        ]

    def _denormalize_serialized_tokens(self, token_ids: List[int]) -> List[int]:
        reserved_token_count = self._reserved_token_count()
        denormalized = []
        for token in token_ids:
            token = int(token)
            if token == self.special_tokens.component_sep_token_id:
                denormalized.append(self.serializer.component_sep_token_id)
            elif token < reserved_token_count:
                raise ValueError(f"Encoded graph contains reserved token id {token}.")
            else:
                denormalized.append(token - reserved_token_count)
        return denormalized

    def _reserved_token_count(self) -> int:
        return max(
            self.special_tokens.pad_token_id,
            self.special_tokens.unk_token_id,
            self.special_tokens.mask_token_id,
            self.special_tokens.cls_token_id,
            self.special_tokens.sep_token_id,
            self.special_tokens.node_start_token_id,
            self.special_tokens.node_end_token_id,
            self.special_tokens.component_sep_token_id,
        ) + 1

    def _special_token_ids(self):
        return set(vars(self.special_tokens).values())

    def _protect_special_tokens(self) -> None:
        if not hasattr(self.bpe, "protected_token_ids"):
            self.bpe.protected_token_ids = set()
        if self.serializer.protocol == "local_refs":
            self.bpe.protected_token_ids.update(self._special_token_ids())

    def _vocabulary_encode(self, token_ids: List[int]) -> List[int]:
        special_ids = set() if self.serializer.protocol == "official" else self._special_token_ids()
        return [
            int(token) if int(token) in special_ids
            else self.vocabulary.get(int(token), self.special_tokens.unk_token_id)
            for token in token_ids
        ]

    def build_mlm_batch(
        self,
        graphs,
        max_length: int,
        mask_prob: float = 0.15,
        seed: int = 0,
    ) -> GraphTokenizerMLMBatch:
        r"""Encode graphs and create a padded GraphTokenizerMLMBatch.

        max_length is the positive final sequence length, including CLS / SEP;
        overlength graphs raise ValueError and are never truncated. mask_prob
        defaults to 0.15 and seed to 0. Delegates to
        build_mlm_batch_from_token_sequences; see that method for mask semantics.
        """
        return self.build_mlm_batch_from_token_sequences(
            [self.encode_graph(graph).input_ids for graph in graphs],
            max_length=max_length,
            mask_prob=mask_prob,
            seed=seed,
        )

    def validate_token_sequences(self, token_sequences, max_length: int) -> None:
        r"""Validate a positive max_length and all final token sequence lengths.

        Raises ValueError if any sequence exceeds the limit. Returns None on
        success; this check does not validate token domains or truncate inputs.
        """
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        if any(len(sequence) > max_length for sequence in token_sequences):
            raise ValueError(
                "Token sequence exceeds max_length; truncation would discard graph structure.")

    def pad_token_sequences(self, token_sequences, max_length: int):
        r"""Right-pad ragged model token_sequences to max_length.

        Returns (input_ids, attention_mask), two Python lists of shape
        (B, max_length); padding uses pad_token_id and mask 0. Raises ValueError
        for nonpositive limits or overlength sequences; no truncation occurs.
        """
        self.validate_token_sequences(token_sequences, max_length=max_length)
        input_ids = [
            [int(token) for token in sequence]
            for sequence in token_sequences
        ]
        attention_mask = [
            [1] * len(sequence) + [0] * (max_length - len(sequence))
            for sequence in input_ids
        ]
        padded_ids = [
            sequence + [self.special_tokens.pad_token_id] * (max_length - len(sequence))
            for sequence in input_ids
        ]
        return padded_ids, attention_mask

    def build_mlm_batch_from_token_sequences(
        self,
        token_sequences,
        max_length: int,
        mask_prob: float = 0.15,
        seed: int = 0,
    ) -> GraphTokenizerMLMBatch:
        r"""Build a padded MLM batch from already encoded model token sequences.

        Parameters
        ----------
        token_sequences: ``iterable[list[int]]``
            Ragged final vocabulary IDs, including any CLS / SEP tokens.
        max_length: int
            Positive padded sequence length; oversized inputs raise ValueError.
        mask_prob: float
            Selection probability for eligible positions. (default: ``0.15``)
        seed: int
            Python masking RNG seed. (default: ``0``)

        Returns
        -------
        GraphTokenizerMLMBatch
            Lists of shape (B, max_length). Selected tokens are replaced by MASK;
            labels retain their IDs, with -100 at other positions. PAD, CLS, SEP
            and the reserved component ID are excluded. Positive mask_prob
            forces one eligible target if none was selected. All-special batches
            can still have zero targets. This helper is separate from the example
            trainer's 80/10/10 MLM corruption policy.
        """
        padded_ids, attention_mask = self.pad_token_sequences(
            token_sequences, max_length=max_length)
        masked_ids, labels = self.mask_token_sequences(
            padded_ids, mask_prob=mask_prob, seed=seed)
        num_mlm_labels = sum(1 for row in labels for value in row if value != -100)
        return GraphTokenizerMLMBatch(
            input_ids=masked_ids,
            attention_mask=attention_mask,
            labels=labels,
            metadata={
                "mask_prob": float(mask_prob),
                "max_length": int(max_length),
                "num_graphs": len(padded_ids),
                "num_mlm_labels": num_mlm_labels,
            },
        )

    def mask_token_sequences(
        self, input_ids: List[List[int]], mask_prob: float, seed: int):
        r"""Mask model input_ids using mask_prob and the given Python RNG seed.

        input_ids is a list of integer lists (B, L_i). Returns two lists of the
        same shapes: masked IDs and labels (-100 except at targets). Uses the
        exclusion / forced-target semantics of build_mlm_batch_from_token_sequences.
        This operates on final model IDs, not raw graph tokens.
        """
        rng = random.Random(seed)
        special_ids = {
            self.special_tokens.pad_token_id,
            self.special_tokens.cls_token_id,
            self.special_tokens.sep_token_id,
            self.special_tokens.component_sep_token_id,
        }
        masked_ids = []
        labels = []
        for sequence in input_ids:
            masked_sequence = []
            label_sequence = []
            for token in sequence:
                token = int(token)
                if token in special_ids or rng.random() >= mask_prob:
                    masked_sequence.append(token)
                    label_sequence.append(-100)
                else:
                    masked_sequence.append(self.special_tokens.mask_token_id)
                    label_sequence.append(token)
            masked_ids.append(masked_sequence)
            labels.append(label_sequence)
        if (
                float(mask_prob) > 0
                and not any(label != -100 for row in labels for label in row)):
            for row_index, sequence in enumerate(input_ids):
                for column_index, token in enumerate(sequence):
                    token = int(token)
                    if token not in special_ids:
                        masked_ids[row_index][column_index] = (
                            self.special_tokens.mask_token_id)
                        labels[row_index][column_index] = token
                        return masked_ids, labels
        return masked_ids, labels

    def _mask_input_ids(self, input_ids: List[List[int]], mask_prob: float, seed: int):
        return self.mask_token_sequences(input_ids, mask_prob=mask_prob, seed=seed)

    def _require_fitted(self) -> None:
        if getattr(self, "schema_version", None) != TOKENIZER_SCHEMA_VERSION:
            raise RuntimeError("Incompatible GraphTokenizer schema; re-fit with the current protocol.")
        if not self._fitted:
            raise RuntimeError("GraphTokenizer must be fit on training graphs before encoding.")
        if getattr(self, "original_token_ids", None) is None:
            raise RuntimeError(
                "GraphTokenizer lacks training original token ids; re-fit or rebuild it before encoding.")
        if getattr(self, "_fit_configuration", None) != self._configuration():
            raise RuntimeError("Incompatible tokenizer protocol or serializer configuration; re-fit.")

    def __call__(self, data: Any):
        r"""Encode data in place and return it.

        data is a supported graph / dict. Adds input_ids, attention_mask,
        serialized_token_ids and graph_tokenizer_metadata. The lists are
        unpadded; exceptions are those of encode_graph.
        """
        encoding = self.encode_graph(data)
        if isinstance(data, dict):
            data["input_ids"] = encoding.input_ids
            data["attention_mask"] = encoding.attention_mask
            data["serialized_token_ids"] = encoding.serialized_token_ids
            data["graph_tokenizer_metadata"] = encoding.metadata
        else:
            setattr(data, "input_ids", encoding.input_ids)
            setattr(data, "attention_mask", encoding.attention_mask)
            setattr(data, "serialized_token_ids", encoding.serialized_token_ids)
            setattr(data, "graph_tokenizer_metadata", encoding.metadata)
        return data
