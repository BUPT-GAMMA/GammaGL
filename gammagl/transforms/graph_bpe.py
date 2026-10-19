import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


MergeRule = Tuple[int, int, int]


@dataclass
class BPECodebook:
    r"""Serializable BPE rules and vocabulary metadata.

    Parameters
    ----------
    merge_rules: list[tuple[int, int, int]]
        Ordered (left, right, merged) rules. Fresh empty list by default.
    vocab_size: int
        Exclusive token ID upper bound, including gaps; not the number of
        observed token types or the final model vocabulary size.
        (default: ``0``)
    metadata: dict
        Fresh empty dict by default; fitted codebooks record backend and
        merge counts. This metadata alone does not bind a graph protocol.

    Notes
    -----
    Produced by :meth:`GraphBPE.fit`; see that class for the official algorithm
    reference. Model IDs and special tokens belong to :class:`GraphTokenizer`.

    Examples
    --------
    >>> from gammagl.transforms import BPECodebook
    >>> codebook = BPECodebook(merge_rules=[(8, 9, 10)], vocab_size=11)
    >>> codebook.merge_rules[0]
    (8, 9, 10)
    """

    merge_rules: List[MergeRule] = field(default_factory=list)
    vocab_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphBPE:
    r"""Byte Pair Encoding for integer graph-token sequences.

    Implements deterministic pair-frequency merging used in the
    `GraphTokenizer paper <https://arxiv.org/abs/2603.11099>`_ and its `official
    GraphTokenizer compression code
    <https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/tree/98343a6b025a48fbb6859cd812a12b81ec3ac3cc/src/algorithms/compression>`_.
    See :class:`GraphTokenizer` for graph construction and model vocabulary
    mapping; BPE inversion alone does not reconstruct a graph.

    Parameters
    ----------
    num_merges: int
        Maximum number of pair merges. (default: ``2000``)
    min_frequency: int
        Minimum corpus pair count; stop when no pair qualifies.
        (default: ``2``)
    backend: str
        "python", "cpp" or "auto". cpp requires the optional native extension;
        auto uses it when available. Pair-count ties choose lexicographically
        smallest pairs in both implementations. (default: ``"python"``)
    protected_token_ids: ``iterable[int]``
        IDs never merged or crossed during training / encoding. None means
        no protection. These are BPE alphabet IDs, not model vocabulary IDs.
        (default: :obj:`None`)
    merge_id_offset: int
        Extra reserved IDs before assigning merged IDs. The official
        GraphTokenizer sets 1 to reserve max(base)+1 as a virtual separator;
        standalone BPE defaults to 0. (default: ``0``)

    Notes
    -----
    fit / batch_encode take a ragged sequence of integer sequences (S_i,).
    encode / decode take one sequence (S,) and return a list[int] of variable
    length. No padding, graph_id grouping or model special tokens are added.

    Raises
    ------
    ValueError
        If fit receives an unsupported backend or decode encounters cyclic
        merge rules. Invalid codebook suffixes also raise ValueError.
    ImportError
        If cpp is requested but the native extension is unavailable.

    Examples
    --------
    >>> from gammagl.transforms import GraphBPE
    >>> bpe = GraphBPE(num_merges=1).fit([[8, 9, 8, 9]])
    >>> bpe.encode([8, 9, 8, 9])
    [10, 10]
    >>> bpe.decode([10, 10])
    [8, 9, 8, 9]
    """

    def __init__(
        self,
        num_merges: int = 2000,
        min_frequency: int = 2,
        backend: str = "python",
        protected_token_ids=None,
        merge_id_offset: int = 0,
    ):
        self.num_merges = int(num_merges)
        self.min_frequency = int(min_frequency)
        self.backend = backend
        self.merge_id_offset = int(merge_id_offset)
        self.protected_token_ids = {
            int(token) for token in (protected_token_ids or ())}
        self.codebook = BPECodebook()

    def fit(self, token_sequences):
        r"""Fit a codebook on ragged integer sequences.

        Parameters
        ----------
        token_sequences: ``iterable of sequences of int``
            Corpus with one sequence (S_i,) per graph. Pair counts never span
            graphs or cross protected tokens. Replaces any existing codebook.

        Returns
        -------
        GraphBPE
            This fitted instance.

        Raises
        ------
        ValueError
            If backend is unsupported.
        ImportError
            If cpp is requested but the native extension is unavailable.
        """
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
        next_token_id = self._initial_vocab_size(original_sequences or sequences)

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
        r"""Merge one sequence using the current rule table.

        Parameters
        ----------
        token_sequence: ``sequence of int``
            One sequence (S,) in the BPE alphabet, before model ID mapping.

        Returns
        -------
        list[int]
            Compressed IDs, applying rules in order and respecting protected IDs.
            An empty codebook acts as identity. No fitting guard or unknown-token
            mapping is applied; :class:`GraphTokenizer` owns that mapping.
        """
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
        r"""Encode a ragged batch without padding.

        Parameters
        ----------
        token_sequences: ``iterable of sequences of int``
            One sequence (S_i,) per graph in the BPE alphabet.

        Returns
        -------
        list[list[int]]
            One compressed sequence per input, equivalent to :meth:`encode`.
        """
        bridge = self._native_bridge()
        if not self.protected_token_ids and bridge is not None and self.codebook.merge_rules:
            return bridge.batch_encode(token_sequences, self.codebook.merge_rules)
        return [self.encode(sequence) for sequence in token_sequences]

    def decode(self, token_sequence):
        r"""Expand merged IDs to base tokens.

        Parameters
        ----------
        token_sequence: ``sequence of int``
            One BPE-encoded sequence. Unmerged IDs pass through unchanged.

        Returns
        -------
        list[int]
            Expanded base tokens. This does not recover graph node identities
            or edge labels omitted by serialization.

        Raises
        ------
        ValueError
            If merge rules contain a cycle.
        """
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
        r"""Save the rule table and metadata.

        Parameters
        ----------
        path: str or ``path-like``
            Destination ending in .json, .pkl or .pickle. The parent must exist.

        Returns
        -------
        None
            Writes merge_rules, vocab_size and metadata. Does not save tokenizer
            state, protected IDs, merge offset, protocol or model vocabulary binding.

        Raises
        ------
        ValueError
            If the suffix is unsupported. File errors propagate.
        """
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
        r"""Load a rule table without reconstructing a graph tokenizer.

        Parameters
        ----------
        path: str or ``path-like``
            Existing .json, .pkl or .pickle file. Pickles must be trusted.
        backend: str
            Execution backend for the new instance. Retained codebook metadata can
            still enable native encoding when available. (default: ``"python"``)

        Returns
        -------
        GraphBPE
            Instance with loaded rules and metadata. Protected IDs and merge offset
            retain constructor defaults. No tokenizer schema validation is implied.

        Raises
        ------
        ValueError
            If the suffix is unsupported. File and parse errors propagate.
        """
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

        initial_vocab_size = self._initial_vocab_size(
            original_sequences or token_sequences)
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

    def _initial_vocab_size(self, sequences: Iterable[List[int]]) -> int:
        return self.merge_id_offset + max(
            self._next_token_id(sequences),
            max(self.protected_token_ids, default=-1) + 1,
        )

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
