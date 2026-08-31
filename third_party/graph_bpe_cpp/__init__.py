from typing import List, Sequence, Tuple


MergeRule = Tuple[int, int, int]


def _load_native():
    try:
        from . import _graph_bpe
    except ImportError:
        return None
    return _graph_bpe


def is_available() -> bool:
    """Return whether the optional native BPE extension is importable."""

    return _load_native() is not None


def backend_name() -> str:
    return "cpp" if is_available() else "python"


def train_bpe(token_sequences, num_merges: int, min_frequency: int, initial_vocab_size=None):
    native = _load_native()
    if native is None:
        raise ImportError("graph_bpe_cpp requires the optional native extension.")
    return _normalize_result(
        native.train_bpe(
            token_sequences,
            int(num_merges),
            int(min_frequency),
            -1 if initial_vocab_size is None else int(initial_vocab_size),
        ), "cpp")


def encode(token_sequence, merge_rules: Sequence[Sequence[int]]) -> List[int]:
    native = _load_native()
    if native is None:
        raise ImportError("graph_bpe_cpp requires the optional native extension.")
    return [int(token) for token in native.encode(token_sequence, merge_rules)]


def batch_encode(token_sequences, merge_rules: Sequence[Sequence[int]]) -> List[List[int]]:
    native = _load_native()
    if native is None:
        raise ImportError("graph_bpe_cpp requires the optional native extension.")
    return [
        [int(token) for token in sequence]
        for sequence in native.batch_encode(token_sequences, merge_rules)
    ]


def _train_bpe_python(token_sequences, num_merges: int, min_frequency: int):
    """Reference implementation used to verify the optional native backend."""
    sequences = [[int(token) for token in sequence] for sequence in token_sequences]
    next_token_id = max((token for sequence in sequences for token in sequence), default=-1) + 1
    merge_rules = []
    for _ in range(int(num_merges)):
        counts = {}
        for sequence in sequences:
            for pair in zip(sequence, sequence[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        pair = min(counts, key=lambda item: (-counts[item], item))
        if counts[pair] < int(min_frequency):
            break
        rule = (*pair, next_token_id)
        merge_rules.append(rule)
        sequences = [_apply_merge(sequence, rule) for sequence in sequences]
        next_token_id += 1
    return {"merge_rules": merge_rules, "vocab_size": next_token_id}


def _apply_merge(sequence, rule):
    left, right, merged = rule
    encoded = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and sequence[index:index + 2] == [left, right]:
            encoded.append(merged)
            index += 2
        else:
            encoded.append(sequence[index])
            index += 1
    return encoded


def _normalize_result(result, backend: str):
    merge_rules = [_as_merge_rule(rule) for rule in result.get("merge_rules", [])]
    return {
        "merge_rules": merge_rules,
        "vocab_size": int(result.get("vocab_size", 0)),
        "metadata": {
            **{key: int(value) for key, value in result.get("metadata", {}).items() if isinstance(value, int)},
            "backend": backend,
        },
    }


def _as_merge_rule(rule) -> MergeRule:
    return tuple(int(value) for value in rule)
