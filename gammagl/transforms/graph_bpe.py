import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


MergeRule = Tuple[int, int, int]


@dataclass
class BPECodebook:
    """Serializable BPE merge table."""

    merge_rules: List[MergeRule] = field(default_factory=list)
    vocab_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphBPE:
    """Minimal public interface for graph-token BPE training and encoding."""

    def __init__(
        self,
        num_merges: int = 2000,
        min_frequency: int = 2,
        backend: str = "python",
        protected_token_ids=None,
    ):
        self.num_merges = int(num_merges)
        self.min_frequency = int(min_frequency)
        self.backend = backend
        self.protected_token_ids = {
            int(token) for token in (protected_token_ids or ())}
        self.codebook = BPECodebook()

    def fit(self, token_sequences):
        sequences = [self._as_int_list(sequence) for sequence in token_sequences]
        training_sequences = self._training_segments(sequences)
        if self.backend == "cpp":
            return self._fit_with_bridge(training_sequences, sequences)
        if self.backend == "auto":
            from third_party import graph_bpe_cpp
            if graph_bpe_cpp.is_available():
                return self._fit_with_bridge(training_sequences, sequences)
        elif self.backend != "python":
            raise ValueError("backend must be one of: python, auto, cpp.")

        return self._fit_python(training_sequences, sequences)

    def _fit_python(self, token_sequences, original_sequences=None):
        sequences = [self._as_int_list(sequence) for sequence in token_sequences]
        merge_rules: List[MergeRule] = []
        next_token_id = self._next_token_id(original_sequences or sequences)

        for _ in range(self.num_merges):
            pair_counts = self._count_pairs(sequences)
            if not pair_counts:
                break

            best_pair, best_count = self._select_best_pair(pair_counts)
            if best_count < self.min_frequency:
                break

            merge_rule = (best_pair[0], best_pair[1], next_token_id)
            merge_rules.append(merge_rule)
            sequences = [self._apply_merge(sequence, merge_rule) for sequence in sequences]
            next_token_id += 1

        self.codebook = BPECodebook(
            merge_rules=merge_rules,
            vocab_size=next_token_id,
            metadata={
                "num_merges_requested": self.num_merges,
                "num_merges_performed": len(merge_rules),
                "min_frequency": self.min_frequency,
                "backend": "python",
            },
        )
        return self

    def encode(self, token_sequence):
        bridge = self._native_bridge()
        pieces = self._split_on_protected(self._as_int_list(token_sequence))
        encoded = []
        for protected, piece in pieces:
            if protected:
                encoded.extend(piece)
            elif bridge is not None and self.codebook.merge_rules:
                encoded.extend(bridge.encode(piece, self.codebook.merge_rules))
            else:
                for merge_rule in self.codebook.merge_rules:
                    piece = self._apply_merge(piece, merge_rule)
                encoded.extend(piece)
        return encoded

    def batch_encode(self, token_sequences):
        bridge = self._native_bridge()
        if not self.protected_token_ids and bridge is not None and self.codebook.merge_rules:
            return bridge.batch_encode(token_sequences, self.codebook.merge_rules)
        return [self.encode(sequence) for sequence in token_sequences]

    def decode(self, token_sequence):
        """Expand BPE merge tokens back to the original token sequence."""
        expansions = {
            int(merged): (int(left), int(right))
            for left, right, merged in self.codebook.merge_rules
        }
        decoded = []
        for token in self._as_int_list(token_sequence):
            decoded.extend(self._expand_token(token, expansions, set()))
        return decoded

    @classmethod
    def _expand_token(cls, token: int, expansions, visiting) -> List[int]:
        if token not in expansions:
            return [token]
        if token in visiting:
            raise ValueError(f"BPE merge rules contain a cycle at token {token}.")
        left, right = expansions[token]
        visiting.add(token)
        expanded = cls._expand_token(left, expansions, visiting) + cls._expand_token(
            right, expansions, visiting)
        visiting.remove(token)
        return expanded

    def _native_bridge(self):
        if self.backend != "cpp" and self.codebook.metadata.get("backend") != "cpp":
            return None
        from third_party import graph_bpe_cpp
        if graph_bpe_cpp.is_available():
            return graph_bpe_cpp
        if self.backend == "cpp":
            raise ImportError(
                "GraphBPE backend='cpp' requires the optional graph_bpe_cpp native extension. "
                "Use backend='auto' or backend='python' to allow fallback.")
        return None

    def save_codebook(self, path) -> None:
        path = Path(path)
        data = {
            "merge_rules": [list(rule) for rule in self.codebook.merge_rules],
            "vocab_size": self.codebook.vocab_size,
            "metadata": dict(self.codebook.metadata),
        }
        if path.suffix.lower() in {".pkl", ".pickle"}:
            with path.open("wb") as handle:
                pickle.dump(data, handle)
            return
        if path.suffix.lower() == ".json":
            with path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False)
            return
        raise ValueError("Codebook path must end with .json, .pkl, or .pickle.")

    @classmethod
    def load_codebook(cls, path, backend: str = "python") -> "GraphBPE":
        path = Path(path)
        if path.suffix.lower() in {".pkl", ".pickle"}:
            with path.open("rb") as handle:
                data = pickle.load(handle)
        elif path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        else:
            raise ValueError("Codebook path must end with .json, .pkl, or .pickle.")

        engine = cls(num_merges=0, backend=backend)
        engine.codebook = BPECodebook(
            merge_rules=[tuple(int(value) for value in rule) for rule in data.get("merge_rules", [])],
            vocab_size=int(data.get("vocab_size", 0)),
            metadata=dict(data.get("metadata", {})),
        )
        return engine

    def _fit_with_bridge(self, token_sequences, original_sequences=None):
        from third_party import graph_bpe_cpp

        if self.backend == "cpp" and not graph_bpe_cpp.is_available():
            raise ImportError(
                "GraphBPE backend='cpp' requires the optional graph_bpe_cpp native extension. "
                "Use backend='auto' or backend='python' to allow fallback."
            )

        initial_vocab_size = self._next_token_id(original_sequences or token_sequences)
        result = graph_bpe_cpp.train_bpe(
            token_sequences,
            num_merges=self.num_merges,
            min_frequency=self.min_frequency,
            initial_vocab_size=initial_vocab_size,
        )
        self.codebook = BPECodebook(
            merge_rules=[tuple(int(value) for value in rule) for rule in result["merge_rules"]],
            vocab_size=int(result["vocab_size"]),
            metadata=dict(result.get("metadata", {})),
        )
        if "backend" not in self.codebook.metadata:
            self.codebook.metadata["backend"] = graph_bpe_cpp.backend_name()
        return self

    def _training_segments(self, sequences):
        if not self.protected_token_ids:
            return sequences
        return [
            piece
            for sequence in sequences
            for protected, piece in self._split_on_protected(sequence)
            if not protected and piece
        ]

    def _split_on_protected(self, sequence):
        if not self.protected_token_ids:
            return [(False, sequence)]
        pieces = []
        current = []
        for token in sequence:
            if token in self.protected_token_ids:
                if current:
                    pieces.append((False, current))
                    current = []
                pieces.append((True, [token]))
            else:
                current.append(token)
        if current:
            pieces.append((False, current))
        return pieces

    @staticmethod
    def _as_int_list(sequence) -> List[int]:
        if hasattr(sequence, "tolist"):
            sequence = sequence.tolist()
        return [int(token) for token in sequence]

    @staticmethod
    def _next_token_id(sequences: Iterable[List[int]]) -> int:
        max_token = -1
        for sequence in sequences:
            if sequence:
                max_token = max(max_token, max(sequence))
        return max_token + 1

    @staticmethod
    def _count_pairs(
        sequences: Iterable[List[int]], protected_token_ids=None,
    ) -> Dict[Tuple[int, int], int]:
        counts: Dict[Tuple[int, int], int] = {}
        protected_token_ids = set(protected_token_ids or ())
        for sequence in sequences:
            for left, right in zip(sequence, sequence[1:]):
                if left in protected_token_ids or right in protected_token_ids:
                    continue
                pair = (left, right)
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _select_best_pair(pair_counts: Dict[Tuple[int, int], int]) -> Tuple[Tuple[int, int], int]:
        best_pair = min(pair_counts.keys(), key=lambda pair: (-pair_counts[pair], pair[0], pair[1]))
        return best_pair, pair_counts[best_pair]

    @staticmethod
    def _apply_merge(sequence: List[int], merge_rule: MergeRule) -> List[int]:
        left, right, new_id = merge_rule
        merged: List[int] = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and sequence[index] == left and sequence[index + 1] == right:
                merged.append(new_id)
                index += 2
            else:
                merged.append(sequence[index])
                index += 1
        return merged


BPEEngine = GraphBPE
