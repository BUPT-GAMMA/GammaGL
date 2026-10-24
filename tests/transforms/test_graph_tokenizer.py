import pickle
import random

import numpy as np
import pytest

from gammagl.data import Graph
from gammagl.transforms.graph_bpe import GraphBPE
from gammagl.transforms.graph_serializer import FrequencyGuidedEulerianSerializer
from gammagl.transforms.graph_tokenizer import GraphTokenizer, GraphTokenizerSpecialTokens


class SimpleGraph:
    def __init__(self, edge_index, x, edge_attr=None, num_nodes=None):
        self.edge_index = edge_index
        self.x = x
        self.edge_attr = edge_attr
        self.num_nodes = len(x) if num_nodes is None else num_nodes


def _edges(edge_index, edge_attr):
    labels = edge_attr if edge_attr is not None else [0] * len(edge_index[0])
    return sorted((min(source, target), max(source, target), label)
                  for source, target, label in zip(edge_index[0], edge_index[1], labels))


def _round_trip(graph):
    serializer = FrequencyGuidedEulerianSerializer(protocol="local_refs").fit([graph])
    result = serializer.serialize(graph)
    restored = serializer.restore_input_metadata(result)
    assert restored["num_nodes"] == graph.num_nodes
    assert restored["x"] == graph.x
    assert _edges(restored["edge_index"], restored["edge_attr"]) == _edges(
        graph.edge_index, graph.edge_attr)


def _relabel(graph, old_to_new):
    new_to_old = {new: old for old, new in old_to_new.items()}
    return SimpleGraph(
        [[old_to_new[index] for index in graph.edge_index[0]],
         [old_to_new[index] for index in graph.edge_index[1]]],
        [graph.x[new_to_old[index]] for index in range(graph.num_nodes)],
        list(graph.edge_attr), graph.num_nodes)


@pytest.mark.parametrize(
    "edge_index,edge_attr",
    [([[0], [1]], [3]), ([[0, 1], [1, 0]], [3, 3])],
    ids=("single_coo", "symmetric_coo"),
)
def test_serializer_accepts_equivalent_undirected_coo(edge_index, edge_attr):
    graph = SimpleGraph(edge_index, [7, 8], edge_attr)
    serializer = FrequencyGuidedEulerianSerializer(protocol="local_refs").fit([graph])

    result = serializer.serialize(graph)

    assert result.metadata["method"] == "feuler"
    assert result.metadata["num_edges_traversed"] == 2
    assert result.token_ids


def test_serializer_normalizes_single_and_symmetric_coo_identically():
    single = SimpleGraph([[0], [1]], [7, 8], [3])
    symmetric = SimpleGraph([[0, 1], [1, 0]], [7, 8], [3, 3])

    assert FrequencyGuidedEulerianSerializer(protocol="local_refs").fit([single]).serialize(single).token_ids == (
        FrequencyGuidedEulerianSerializer(protocol="local_refs").fit([symmetric]).serialize(symmetric).token_ids)


@pytest.mark.parametrize(
    "graph",
    [
        SimpleGraph([[0, 1], [1, 2]], [4, 9, 4], [7, 8]),
        SimpleGraph([[0, 1, 2], [1, 2, 0]], [1, 1, 1], [2, 3, 4]),
        SimpleGraph([[0, 2], [1, 3]], [1, 2, 3, 4], [5, 6]),
        SimpleGraph([[0], [1]], [1, 2, 3], [5], num_nodes=3),
    ],
    ids=("path", "cycle", "disconnected", "isolated_node"),
)
def test_serializer_round_trip(graph):
    _round_trip(graph)


def test_serializer_permutation_regression():
    graph = SimpleGraph(
        [[0, 1, 1, 3], [1, 2, 3, 4]], [9, 1, 4, 7, 4], [3, 2, 5, 6])
    serializer = FrequencyGuidedEulerianSerializer(protocol="local_refs").fit([graph])
    expected = serializer.serialize(graph).token_ids
    permutation = list(range(graph.num_nodes))
    random.Random(7).shuffle(permutation)
    relabeled = _relabel(graph, dict(enumerate(permutation)))

    assert serializer.serialize(relabeled).token_ids == expected
    _round_trip(relabeled)


def test_missing_node_features_match_explicit_zero_labels_in_full_input():
    unlabeled = SimpleGraph([[0, 1], [1, 2]], None, [4, 5], num_nodes=3)
    zero_labeled = SimpleGraph([[0, 1], [1, 2]], [0, 0, 0], [4, 5])
    relabeled = SimpleGraph([[2, 0], [0, 1]], None, [4, 5], num_nodes=3)
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs"), bpe=GraphBPE(num_merges=2, min_frequency=2)).fit(
        [zero_labeled])

    original = tokenizer.encode_graph(unlabeled)
    for graph in (zero_labeled, relabeled):
        encoded = tokenizer.encode_graph(graph)
        assert encoded.serialized_token_ids == original.serialized_token_ids
        assert encoded.input_ids == original.input_ids
    assert tokenizer.decode_graph(original)["x"] == [0, 0, 0]
    assert tokenizer.decode_graph(original.input_ids)["x"] == [0, 0, 0]
    assert original.metadata["serializer"]["local_to_original"] != (
        tokenizer.encode_graph(relabeled).metadata["serializer"]["local_to_original"])


@pytest.mark.parametrize("edge_index,num_nodes,expected_edges", [
    ([[0], [1]], 3, [(0, 1, 0)]),
    ([[], []], 0, []),
], ids=("isolated_node", "empty_graph"))
def test_missing_node_features_restore_default_labels_and_structure(
        edge_index, num_nodes, expected_edges):
    graph = SimpleGraph(edge_index, None, num_nodes=num_nodes)
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs")).fit([graph])
    encoded = tokenizer.encode_graph(graph)

    for restored in (tokenizer.decode_graph(encoded),
                     tokenizer.decode_graph(encoded.input_ids)):
        assert restored["num_nodes"] == num_nodes
        assert restored["x"] == [0] * num_nodes
        assert _edges(restored["edge_index"], restored["edge_attr"]) == expected_edges


def test_explicit_integer_node_labels_are_preserved():
    graph = SimpleGraph([[0], [1]], [7, -2], [3])
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs")).fit([graph])

    assert tokenizer.decode_graph(tokenizer.encode_graph(graph))["x"] == [7, -2]


@pytest.mark.parametrize("graph,permutation", [
    (SimpleGraph([[0, 1], [1, 2]], [10, 20, 30], [1, 2]), {0: 2, 1: 1, 2: 0}),
    (SimpleGraph([[0, 1, 2], [1, 2, 3]], [11, 22, 33, 44], [1, 2, 3]),
     {0: 3, 1: 2, 2: 1, 3: 0}),
    (SimpleGraph([[0, 1, 2, 3], [1, 2, 3, 0]], [11, 22, 33, 44], [1, 2, 3, 4]),
     {0: 2, 1: 0, 2: 3, 3: 1}),
    (SimpleGraph([[0, 2], [1, 3]], [11, 22, 33, 44], [1, 2]),
     {0: 3, 1: 2, 2: 1, 3: 0}),
    (SimpleGraph([[0], [1]], [11, 22, 33], [1]), {0: 2, 1: 1, 2: 0}),
], ids=("three_node", "labeled_path", "labeled_cycle", "disconnected", "isolated"))
def test_full_serialized_tokens_and_input_ids_match_under_relabeling(graph, permutation):
    relabeled = _relabel(graph, permutation)
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs"), bpe=GraphBPE(num_merges=2, min_frequency=2)).fit([graph])
    original = tokenizer.encode_graph(graph)
    permuted = tokenizer.encode_graph(relabeled)

    assert original.serialized_token_ids == permuted.serialized_token_ids
    assert original.input_ids == permuted.input_ids
    assert tokenizer.serializer.serialize(graph).token_ids == (
        tokenizer.serializer.serialize(relabeled).token_ids)


def test_corresponding_explicit_starts_match_full_input_ids():
    graph = SimpleGraph([[0, 1], [1, 2]], [10, 20, 30], [1, 2])
    permutation = {0: 2, 1: 1, 2: 0}
    relabeled = _relabel(graph, permutation)
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs")).fit([graph, relabeled])

    assert tokenizer.encode_graph(graph, start_node=0).input_ids == (
        tokenizer.encode_graph(relabeled, start_node=permutation[0]).input_ids)


def test_multiple_start_nodes_follow_local_traversal_order_under_relabeling():
    graph = SimpleGraph([[0, 1], [1, 2]], [10, 20, 30], [1, 2])
    permutation = {0: 2, 1: 1, 2: 0}
    relabeled = _relabel(graph, permutation)
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs")).fit([graph, relabeled], num_realizations=2)
    starts = tokenizer.realization_start_nodes(graph, 2)
    relabeled_starts = tokenizer.realization_start_nodes(relabeled, 2)

    assert relabeled_starts == [permutation[start] for start in starts]
    assert [tokenizer.encode_graph(graph, start_node=start).input_ids for start in starts] == [
        tokenizer.encode_graph(relabeled, start_node=start).input_ids
        for start in relabeled_starts]


def test_tokenizer_result_restores_original_ids_but_stream_restores_local_ids():
    graph = SimpleGraph([[2, 1], [1, 0]], [30, 20, 10], [1, 2])
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs")).fit([graph])
    result = tokenizer.encode_graph(graph)
    original = tokenizer.decode_graph(result)
    local = tokenizer.decode_graph(result.input_ids)

    assert original["x"] == graph.x
    assert _edges(original["edge_index"], original["edge_attr"]) == (
        _edges(graph.edge_index, graph.edge_attr))
    assert local["x"] == [10, 20, 30]
    assert local["x"] != original["x"]


@pytest.mark.parametrize("graph", [
    SimpleGraph([[0, 2], [1, 3]], [11, 22, 33, 44], [1, 2]),
    SimpleGraph([[0], [1]], [11, 22, 33], [1]),
    SimpleGraph([[], []], [], [], num_nodes=0),
], ids=("components", "isolated", "zero_nodes"))
def test_tokenizer_recovery_handles_components_isolates_and_empty_graph(graph):
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs")).fit([graph])
    result = tokenizer.encode_graph(graph)
    restored = tokenizer.decode_graph(result)
    stream_only = tokenizer.decode_graph(result.input_ids)

    assert restored["num_nodes"] == stream_only["num_nodes"] == graph.num_nodes
    assert restored["x"] == graph.x
    assert _edges(restored["edge_index"], restored["edge_attr"]) == (
        _edges(graph.edge_index, graph.edge_attr))


def test_full_result_restores_held_out_labels_even_when_input_uses_unk():
    train = SimpleGraph([[0], [1]], [1, 2], [1])
    held_out = SimpleGraph([[0], [1]], [101, 2], [1])
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs")).fit([train])
    result = tokenizer.encode_graph(held_out)

    assert tokenizer.special_tokens.unk_token_id in result.input_ids
    assert tokenizer.decode_graph(result)["x"] == [101, 2]


@pytest.mark.parametrize("graph", [
    SimpleGraph([[0, 1], [1, 2]], [10, 20, 30], [1, 2]),
    SimpleGraph([[0, 2], [1, 3]], [11, 22, 33, 44], [1, 2]),
    SimpleGraph([[0], [1]], [11, 22, 33], [1]),
    SimpleGraph([[], []], [], [], num_nodes=0),
], ids=("path", "components", "isolated", "zero_nodes"))
def test_local_references_and_both_recovery_contracts(graph):
    serializer = FrequencyGuidedEulerianSerializer(protocol="local_refs").fit([graph])
    result = serializer.serialize(graph)
    local = serializer.deserialize(result.token_ids)
    original = serializer.restore_input_metadata(result)

    assert result.metadata["protocol_version"] > 2
    assert sorted(result.metadata["local_to_original"]) == list(range(graph.num_nodes))
    assert original["num_nodes"] == graph.num_nodes
    assert original["x"] == graph.x
    assert _edges(original["edge_index"], original["edge_attr"]) == (
        _edges(graph.edge_index, graph.edge_attr))
    assert local["num_nodes"] == graph.num_nodes
    assert sorted(local["x"]) == sorted(graph.x)
    original_to_local = {original_id: local_id for local_id, original_id in
                         enumerate(result.metadata["local_to_original"])}
    assert _edges(local["edge_index"], local["edge_attr"]) == (
        sorted((min(original_to_local[src], original_to_local[dst]),
                max(original_to_local[src], original_to_local[dst]), label)
               for src, dst, label in zip(graph.edge_index[0], graph.edge_index[1], graph.edge_attr)))
    references = [token // 3 for token in result.token_ids
                  if token >= 0 and token % 3 == 2]
    assert set(references) == set(range(graph.num_nodes))
    if graph.edge_index[0]:
        assert len(references) > graph.num_nodes


def test_old_pickled_tokenizer_and_result_are_rejected():
    graph = SimpleGraph([[0], [1]], [10, 20], [1])
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs")).fit([graph])
    old = pickle.loads(pickle.dumps(tokenizer))
    old.schema_version = 2
    with pytest.raises(RuntimeError, match="protocol|schema"):
        old.encode_graph(graph)

    result = tokenizer.encode_graph(graph)
    result.metadata["tokenizer_schema_version"] = 2
    with pytest.raises(ValueError, match="protocol|schema"):
        tokenizer.decode_graph(result)

    serialized = tokenizer.serializer.serialize(graph)
    serialized.metadata["protocol_version"] = 2
    with pytest.raises(ValueError, match="protocol"):
        tokenizer.serializer.deserialize(serialized)


def test_previous_unlabeled_tokenizer_schema_cannot_encode_new_default_labels():
    graph = SimpleGraph([[0], [1]], None, num_nodes=2)
    old = pickle.loads(pickle.dumps(GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs")).fit([graph])))
    old.schema_version = 3

    with pytest.raises(RuntimeError, match="protocol|schema"):
        old.encode_graph(graph)


def test_signature_tie_can_leave_full_input_dependent_on_numbering():
    edges = [(0, 1), (0, 2), (0, 4), (1, 2), (1, 3)]
    graph = SimpleGraph([[src for src, _ in edges], [dst for _, dst in edges]],
                        [1] * 5, [1] * 5)
    permutation = {0: 1, 1: 2, 2: 3, 3: 0, 4: 4}
    relabeled = _relabel(graph, permutation)
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs")).fit([graph, relabeled], num_realizations=2)
    signature = tokenizer.serializer._read_graph(graph)["node_signatures"]

    assert signature[0] == signature[1]
    assert tokenizer.encode_graph(graph).serialized_token_ids != (
        tokenizer.encode_graph(relabeled).serialized_token_ids)
    assert tokenizer.encode_graph(graph).input_ids != tokenizer.encode_graph(relabeled).input_ids
    starts = tokenizer.realization_start_nodes(graph, 2)
    relabeled_starts = tokenizer.realization_start_nodes(relabeled, 2)
    assert relabeled_starts != [permutation[start] for start in starts]


def test_component_sort_tie_can_leave_full_input_dependent_on_numbering():
    graph = SimpleGraph([[0, 1, 2, 4, 5, 4], [1, 2, 3, 5, 6, 6]],
                        [1] * 7, [1] * 6)
    permutation = {0: 3, 1: 4, 2: 5, 3: 6, 4: 0, 5: 1, 6: 2}
    relabeled = _relabel(graph, permutation)
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs")).fit([graph, relabeled])

    assert tokenizer.serializer.serialize(graph).token_ids != (
        tokenizer.serializer.serialize(relabeled).token_ids)
    assert tokenizer.encode_graph(graph).input_ids != tokenizer.encode_graph(relabeled).input_ids


def test_equal_structure_signatures_do_not_prove_nodes_interchangeable():
    # A 4-cycle and a triangle: every vertex has identical one-hop refinement.
    graph = SimpleGraph([[0, 1, 2, 0, 4, 5, 4], [1, 2, 3, 3, 5, 6, 6]],
                        [1] * 7, [1] * 7)
    serializer = FrequencyGuidedEulerianSerializer(protocol="local_refs")
    signatures = serializer._read_graph(graph)["node_signatures"]

    assert len(set(signatures.values())) == 1
    assert len(serializer._connected_components(graph.num_nodes,
               serializer._read_graph(graph)["arcs"])) == 2


@pytest.mark.parametrize(
    "graph,operation,error",
    [
        (SimpleGraph([[0], [0]], [4], [1]), "serialize", "self-loops"),
        (SimpleGraph([[0, 0], [1, 1]], [4, 5], [1, 1]), "fit", "Parallel edges"),
        (SimpleGraph([[0], [1]], [4], [1], num_nodes=2), "fit", "x must contain exactly"),
    ],
    ids=("self_loop", "parallel_edge", "bad_node_features"),
)
def test_serializer_rejects_invalid_graphs(graph, operation, error):
    serializer = FrequencyGuidedEulerianSerializer(protocol="local_refs")

    with pytest.raises(ValueError, match=error):
        getattr(serializer, operation)([graph] if operation == "fit" else graph)


def test_serializer_rejects_explicit_directed_graphs():
    graph = {"edge_index": [[0], [1]], "x": [4, 5], "edge_attr": [1], "directed": True}

    with pytest.raises(ValueError, match="directed graph semantics"):
        FrequencyGuidedEulerianSerializer(protocol="local_refs").serialize(graph)


def test_serializer_accepts_explicit_undirected_native_graph_with_single_coo():
    serializer = FrequencyGuidedEulerianSerializer(protocol="local_refs")
    graph = Graph(
        edge_index=np.array([[0], [1]]),
        x=np.array([4, 5]),
        edge_attr=np.array([1]),
        directed=False,
        to_tensor=False,
    )

    result = serializer.serialize(graph)
    dictionary_result = serializer.serialize(
        {"edge_index": [[0], [1]], "x": [4, 5], "edge_attr": [1]})

    assert result.token_ids == dictionary_result.token_ids


def test_serializer_accepts_native_graph_with_single_coo_without_direction_declaration():
    graph = Graph(
        edge_index=np.array([[0], [1]]),
        x=np.array([4, 5]),
        edge_attr=np.array([1]),
        to_tensor=False,
    )

    assert FrequencyGuidedEulerianSerializer(protocol="local_refs").serialize(graph).token_ids


@pytest.mark.parametrize(
    "graph",
    [
        {"edge_index": [[0], [1]], "x": [1.9, 2], "edge_attr": [3]},
        {"edge_index": [[0], [1]], "x": [1, 2], "edge_attr": [3.9]},
    ],
    ids=("node_label", "edge_label"),
)
def test_serializer_rejects_non_integer_labels(graph):
    with pytest.raises(ValueError, match="labels must be integers"):
        FrequencyGuidedEulerianSerializer(protocol="local_refs").serialize(graph)


@pytest.mark.parametrize("separator", [0, 7, -1.5, True])
def test_serializer_rejects_component_separators_outside_negative_integer_domain(separator):
    with pytest.raises(ValueError, match="component_sep_token_id"):
        FrequencyGuidedEulerianSerializer(protocol="local_refs", component_sep_token_id=separator)


def test_bpe_fits_encodes_and_builds_a_deterministic_codebook(tmp_path):
    sequences = [[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 3]]
    first = GraphBPE(num_merges=2, min_frequency=2).fit(sequences)
    second = GraphBPE(num_merges=2, min_frequency=2).fit(list(reversed(sequences)))

    assert first.codebook.merge_rules == [(1, 2, 4), (4, 4, 5)]
    assert first.codebook == second.codebook
    assert first.encode([1, 2, 1, 2, 3]) == [5, 3]
    path = tmp_path / "codebook.json"
    first.save_codebook(path)
    assert GraphBPE.load_codebook(path).encode([1, 2, 1, 2]) == [5]


def test_bpe_keeps_protected_tokens_out_of_merges():
    bpe = GraphBPE(num_merges=2, min_frequency=2, protected_token_ids={7})
    bpe.fit([[1, 2, 7, 1, 2], [1, 2, 7, 1, 2]])

    assert all(7 not in rule[:2] for rule in bpe.codebook.merge_rules)
    assert 7 in bpe.encode([1, 2, 7, 1, 2])


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_bpe_merge_ids_do_not_collide_with_absent_protected_tokens(backend):
    if backend == "cpp":
        from third_party import graph_bpe_cpp

        if not graph_bpe_cpp.is_available():
            pytest.skip("graph_bpe_cpp native extension is unavailable")

    bpe = GraphBPE(
        num_merges=5,
        min_frequency=2,
        backend=backend,
        protected_token_ids={4},
    ).fit([[0, 1] * 4] * 2)

    assert all(merged != 4 for _, _, merged in bpe.codebook.merge_rules)
    assert bpe.decode(bpe.encode([4])) == [4]


def test_cpp_bpe_matches_python_when_extension_is_available():
    from third_party import graph_bpe_cpp

    if not graph_bpe_cpp.is_available():
        pytest.skip("graph_bpe_cpp native extension is unavailable")
    sequences = [[1, 2, 7, 1, 2], [1, 2, 7, 1, 2], [1, 2]]
    python_bpe = GraphBPE(2, 2, backend="python", protected_token_ids={7}).fit(sequences)
    cpp_bpe = GraphBPE(2, 2, backend="cpp", protected_token_ids={7}).fit(sequences)

    assert cpp_bpe.codebook.merge_rules == python_bpe.codebook.merge_rules
    assert cpp_bpe.encode(sequences[0]) == python_bpe.encode(sequences[0])


def test_tokenizer_encodes_special_tokens_and_pads():
    graph = SimpleGraph([[0], [1]], [1, 2], [5])
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs"), bpe=GraphBPE(num_merges=1, min_frequency=2)).fit([graph, graph])

    encoded = tokenizer.encode_graph(graph)
    padded, attention = tokenizer.pad_token_sequences([encoded.input_ids], max_length=12)

    assert encoded.input_ids[0] == tokenizer.special_tokens.cls_token_id
    assert encoded.input_ids[-1] == tokenizer.special_tokens.sep_token_id
    assert attention[0][:len(encoded.input_ids)] == [1] * len(encoded.input_ids)
    assert padded[0][-1] == tokenizer.special_tokens.pad_token_id


def test_tokenizer_does_not_expand_train_vocabulary_for_held_out_tokens():
    train = SimpleGraph([[0], [1]], [1, 2], [5])
    held_out = SimpleGraph([[0], [1]], [101, 102], [5])
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs"), bpe=GraphBPE(num_merges=1, min_frequency=2)).fit([train, train])
    vocabulary = dict(tokenizer.vocabulary)

    encoded = tokenizer.encode_graph(held_out)

    assert tokenizer.vocabulary == vocabulary
    assert tokenizer.special_tokens.unk_token_id in encoded.input_ids


def _token_collision_graphs():
    train = {"edge_index": [[0], [1]], "x": [1, 2], "edge_attr": [1], "num_nodes": 2}
    return train, {**train, "x": [1, 3]}


def test_tokenizer_maps_unknown_raw_tokens_to_unk_before_bpe_merges():
    train, held_out = _token_collision_graphs()
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs")).fit([train, train])
    held_out_tokens = tokenizer._normalize_serialized_tokens(
        tokenizer.serializer.serialize(held_out).token_ids)
    collision_id = next(token for token in held_out_tokens
                        if token in {rule[2] for rule in tokenizer.bpe.codebook.merge_rules})

    encoded = tokenizer.encode_graph(held_out)

    assert tokenizer.special_tokens.unk_token_id in encoded.input_ids
    assert encoded.input_ids == [3, 13, 11, 1, 17, 4]
    assert collision_id not in tokenizer.original_token_ids


def test_tokenizer_unknown_tokens_and_special_tokens_are_bpe_boundaries():
    train, _ = _token_collision_graphs()
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs")).fit([train, train])
    left, right, merged = tokenizer.bpe.codebook.merge_rules[0]
    tokenizer.bpe.codebook.merge_rules.extend([
        (merged, tokenizer.special_tokens.unk_token_id, merged + 100),
        (merged, tokenizer.special_tokens.mask_token_id, merged + 101),
    ])

    unknown = max(tokenizer.original_token_ids) + 1000
    encoded_unknown = tokenizer.encode_tokens([left, right, unknown, left, right])
    encoded_masked = tokenizer.encode_tokens([
        left, right, tokenizer.special_tokens.mask_token_id, left, right])

    assert tokenizer.special_tokens.unk_token_id in tokenizer.bpe.protected_token_ids
    assert tokenizer.special_tokens.mask_token_id in tokenizer.bpe.protected_token_ids
    assert encoded_unknown == [3, 13, 1, 13, 4]
    assert encoded_masked == [3, 13, tokenizer.special_tokens.mask_token_id, 13, 4]


@pytest.mark.parametrize(
    "special_tokens",
    [GraphTokenizerSpecialTokens(), GraphTokenizerSpecialTokens(pad_token_id=1, unk_token_id=0)],
    ids=("bert", "gte"),
)
def test_tokenizer_uses_configured_unk_for_non_colliding_unknown_tokens(special_tokens):
    train, held_out = _token_collision_graphs()
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs"), special_tokens=special_tokens).fit([train, train])
    non_colliding = {**held_out, "x": [1, 101]}

    encoded = tokenizer.encode_graph(non_colliding)

    assert special_tokens.unk_token_id in encoded.input_ids
    assert tokenizer.encode_graph(train).input_ids == [3, 18, 4]


def test_tokenizer_batch_encoding_and_persistence_reuse_original_token_ids():
    train, held_out = _token_collision_graphs()
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs")).fit([train, train])
    restored = pickle.loads(pickle.dumps(tokenizer))

    single = [tokenizer.encode_graph(graph).input_ids for graph in (train, held_out)]
    batch = [result.input_ids for result in tokenizer.batch_encode_graphs([train, held_out])]

    assert batch == single
    assert restored.original_token_ids == tokenizer.original_token_ids
    assert restored.encode_graph(held_out).input_ids == tokenizer.encode_graph(held_out).input_ids
    del restored.original_token_ids
    with pytest.raises(RuntimeError, match="re-fit"):
        restored.encode_graph(train)


def test_tokenizer_rejects_unfitted_and_overlong_sequences():
    graph = SimpleGraph([[0], [1]], [1, 2], [5])
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs"))

    with pytest.raises(RuntimeError, match="must be fit"):
        tokenizer.encode_graph(graph)
    with pytest.raises(ValueError, match="exceeds max_length"):
        tokenizer.pad_token_sequences([[3, 8, 9, 4]], max_length=3)


def test_tokenizer_validates_token_sequence_lengths_without_truncating():
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs"))

    assert tokenizer.validate_token_sequences([[3, 8, 9, 10, 11, 12, 13, 4]], max_length=8) is None
    with pytest.raises(ValueError, match="exceeds max_length"):
        tokenizer.validate_token_sequences([[3, 8, 9, 10, 11, 12, 13, 14, 4]], max_length=8)
