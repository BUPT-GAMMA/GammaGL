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
    """Tokenized graph sequence and lightweight provenance."""

    input_ids: List[int]
    attention_mask: List[int]
    serialized_token_ids: List[int]
    metadata: Dict[str, Any]


@dataclass
class GraphTokenizerMLMBatch:
    """Padded masked-token batch for MLM pretraining."""

    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    labels: List[List[int]]
    metadata: Dict[str, Any]


class GraphTokenizer(BaseTransform):
    """Composes graph serialization and BPE into token ids.

    With the default Feuler serializer, inputs are limited to simple
    undirected graphs without self-loops or parallel edges.  A custom
    serializer owns any broader graph-domain contract.
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
        """Bind artifacts to the actual protocol, fitted statistics, BPE and vocabulary."""
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
        encoded = self._encode_token_sequences([token_ids])[0]
        if not self.add_special_tokens:
            return encoded
        return [self.special_tokens.cls_token_id, *encoded, self.special_tokens.sep_token_id]

    @property
    def max_token_id(self) -> int:
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
        required_vocab_size = self.max_token_id + 1
        if int(vocab_size) < required_vocab_size:
            raise ValueError(
                f"Model vocab_size={vocab_size} is smaller than the tokenizer "
                f"requirement {required_vocab_size}.")

    def encode_graph(
            self, graph: Any, start_node: int | None = None
    ) -> GraphTokenizationResult:
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
        """Space starts over first-visit order for the default serializer."""
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
        """Reconstruct only local_refs; official streams have no node identity."""
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
        return self.build_mlm_batch_from_token_sequences(
            [self.encode_graph(graph).input_ids for graph in graphs],
            max_length=max_length,
            mask_prob=mask_prob,
            seed=seed,
        )

    def validate_token_sequences(self, token_sequences, max_length: int) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        if any(len(sequence) > max_length for sequence in token_sequences):
            raise ValueError(
                "Token sequence exceeds max_length; truncation would discard graph structure.")

    def pad_token_sequences(self, token_sequences, max_length: int):
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
