from dataclasses import dataclass
import hashlib
import random
from typing import Any, Dict, List

from .base_transform import BaseTransform
from .graph_bpe import GraphBPE
from .graph_serializer import FrequencyGuidedEulerianSerializer


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
    ):
        self.serializer = serializer or FrequencyGuidedEulerianSerializer()
        self.bpe = bpe or GraphBPE()
        self.special_tokens = special_tokens or GraphTokenizerSpecialTokens()
        if not hasattr(self.bpe, "protected_token_ids"):
            self.bpe.protected_token_ids = set()
        self.bpe.protected_token_ids.add(
            self.special_tokens.component_sep_token_id)
        self.add_special_tokens = bool(add_special_tokens)
        self._fitted = False
        self.vocabulary = {}
        self.id_to_vocabulary_token = {}
        self.fit_graph_ids_hash = None
        self.schema_version = 1

    def fit(self, graphs, graph_ids=None):
        graphs = list(graphs)
        if not graphs:
            raise ValueError("Cannot fit GraphTokenizer on an empty training corpus.")
        self.fit_graph_ids_hash = None
        if graph_ids is not None:
            graph_ids = list(graph_ids)
            if len(graph_ids) != len(graphs):
                raise ValueError("graph_ids must align with the training graphs.")
            self.fit_graph_ids_hash = hashlib.sha256(
                repr(tuple(graph_ids)).encode("utf-8")).hexdigest()
        self.serializer.fit(graphs)
        serialized_sequences = [
            self._normalize_serialized_tokens(self.serializer.serialize(graph).token_ids)
            for graph in graphs
        ]
        self.bpe.fit(serialized_sequences)
        # Match the official protocol: BPE IDs are an intermediate alphabet,
        # then a frozen train-built contiguous vocabulary maps them to model IDs.
        # This permits inductive val/test encoding through UNK without changing
        # BPE rules or admitting evaluation graphs into vocabulary fitting.
        base_ids = {int(token) for sequence in serialized_sequences for token in sequence}
        merge_ids = {int(rule[2]) for rule in self.bpe.codebook.merge_rules}
        special_ids = set(vars(self.special_tokens).values())
        token_ids = sorted((base_ids | merge_ids) - special_ids)
        start = self._reserved_token_count()
        self.vocabulary = {token: start + index for index, token in enumerate(token_ids)}
        self.id_to_vocabulary_token = {
            vocab_id: token for token, vocab_id in self.vocabulary.items()}
        self._fitted = True
        return self

    def encode_tokens(self, token_ids: List[int]) -> List[int]:
        encoded = self._vocabulary_encode(self.bpe.encode(token_ids))
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

    def encode_graph(self, graph: Any) -> GraphTokenizationResult:
        self._require_fitted()
        serialized = self.serializer.serialize(graph)
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

    def decode_graph(self, encoding) -> Dict[str, Any]:
        """Invert BPE and reconstruct a graph from an encoded token sequence."""
        self._require_fitted()
        if isinstance(encoding, GraphTokenizationResult):
            input_ids = encoding.input_ids
        else:
            input_ids = list(encoding)
        if self.add_special_tokens:
            if (
                    len(input_ids) < 2
                    or input_ids[0] != self.special_tokens.cls_token_id
                    or input_ids[-1] != self.special_tokens.sep_token_id):
                raise ValueError("Encoded graph must begin with CLS and end with SEP tokens.")
            input_ids = input_ids[1:-1]
        serialized = self.bpe.decode([
            self.id_to_vocabulary_token.get(token, token)
            for token in input_ids
        ])
        return self.serializer.deserialize(
            self._denormalize_serialized_tokens(serialized))

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
        encoded_sequences = [
            self._vocabulary_encode(sequence)
            for sequence in self.bpe.batch_encode(serialized_token_ids)
        ]
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

    def _vocabulary_encode(self, token_ids: List[int]) -> List[int]:
        special_ids = set(vars(self.special_tokens).values())
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

    def pad_token_sequences(self, token_sequences, max_length: int):
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        if any(len(sequence) > max_length for sequence in token_sequences):
            raise ValueError(
                "Token sequence exceeds max_length; truncation would discard graph structure.")
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
        if not self._fitted:
            raise RuntimeError("GraphTokenizer must be fit on training graphs before encoding.")

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
