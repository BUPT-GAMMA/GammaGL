import importlib.util
import random
import sys
import types
from pathlib import Path

import pytest

def _load_graph_serializer_module():
    module_path = Path(__file__).resolve().parents[2] / "gammagl" / "transforms" / "graph_serializer.py"
    spec = importlib.util.spec_from_file_location("graph_serializer_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_graph_bpe_module():
    module_path = Path(__file__).resolve().parents[2] / "gammagl" / "transforms" / "graph_bpe.py"
    spec = importlib.util.spec_from_file_location("graph_bpe_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_graph_tokenizer_module():
    root = Path(__file__).resolve().parents[2]
    gammagl_package = sys.modules.setdefault("gammagl", types.ModuleType("gammagl"))
    gammagl_package.__path__ = [str(root / "gammagl")]
    transforms_package = sys.modules.setdefault("gammagl.transforms", types.ModuleType("gammagl.transforms"))
    transforms_package.__path__ = [str(root / "gammagl" / "transforms")]
    module_path = Path(__file__).resolve().parents[2] / "gammagl" / "transforms" / "graph_tokenizer.py"
    spec = importlib.util.spec_from_file_location("gammagl.transforms.graph_tokenizer_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_graph_tokenizer_public_interfaces_are_importable():
    graph_bpe = _load_graph_bpe_module()
    graph_serializer = _load_graph_serializer_module()
    graph_tokenizer = _load_graph_tokenizer_module()
    BPECodebook = graph_bpe.BPECodebook
    GraphBPE = graph_bpe.GraphBPE
    FrequencyGuidedEulerianSerializer = graph_serializer.FrequencyGuidedEulerianSerializer
    GraphTokenizer = graph_tokenizer.GraphTokenizer
    GraphTokenizationResult = graph_tokenizer.GraphTokenizationResult
    GraphTokenizerMLMBatch = graph_tokenizer.GraphTokenizerMLMBatch
    from third_party.graph_bpe_cpp import is_available

    serializer = FrequencyGuidedEulerianSerializer()
    bpe = GraphBPE()
    tokenizer = GraphTokenizer(serializer=serializer, bpe=bpe)
    codebook = BPECodebook(merge_rules=[], vocab_size=0)

    assert serializer.name == "feuler"
    assert bpe.backend == "python"
    assert tokenizer.serializer is serializer
    assert tokenizer.bpe is bpe
    assert codebook.merge_rules == []
    assert GraphTokenizationResult(input_ids=[], attention_mask=[], serialized_token_ids=[], metadata={}).input_ids == []
    assert GraphTokenizerMLMBatch(input_ids=[], attention_mask=[], labels=[], metadata={}).labels == []
    assert isinstance(is_available(), bool)


class SimpleGraph:
    def __init__(self, edge_index, x, edge_attr=None, num_nodes=None):
        self.edge_index = edge_index
        self.x = x
        self.edge_attr = edge_attr
        self.num_nodes = num_nodes if num_nodes is not None else len(x)


def test_feuler_fit_counts_node_edge_node_patterns():
    FrequencyGuidedEulerianSerializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer

    graph = SimpleGraph(
        edge_index=[[0, 1, 1], [1, 0, 2]],
        x=[7, 8, 9],
        edge_attr=[3, 3, 4],
    )

    serializer = FrequencyGuidedEulerianSerializer()
    serializer.fit([graph])

    assert serializer.frequency_map == {
        (7, 3, 8): 1,
        (8, 4, 9): 1,
    }


def test_single_vs_symmetric_coo_same_frequency():
    FrequencyGuidedEulerianSerializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer
    single_direction = SimpleGraph(
        edge_index=[[0], [1]], x=[7, 8], edge_attr=[3])
    symmetric = SimpleGraph(
        edge_index=[[0, 1], [1, 0]], x=[7, 8], edge_attr=[3, 3])

    single_serializer = FrequencyGuidedEulerianSerializer().fit([single_direction])
    symmetric_serializer = FrequencyGuidedEulerianSerializer().fit([symmetric])

    assert single_serializer.frequency_map == symmetric_serializer.frequency_map


def test_single_vs_symmetric_coo_same_serialization():
    FrequencyGuidedEulerianSerializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer
    single_direction = SimpleGraph(
        edge_index=[[0], [1]], x=[7, 8], edge_attr=[3])
    symmetric = SimpleGraph(
        edge_index=[[0, 1], [1, 0]], x=[7, 8], edge_attr=[3, 3])

    single_serializer = FrequencyGuidedEulerianSerializer().fit([single_direction])
    symmetric_serializer = FrequencyGuidedEulerianSerializer().fit([symmetric])

    assert single_serializer.serialize(single_direction).token_ids == (
        symmetric_serializer.serialize(symmetric).token_ids)


def test_invalid_node_feature_length_raises():
    serializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer()
    graph = SimpleGraph(
        edge_index=[[0], [1]], x=[7], edge_attr=[3], num_nodes=2)

    with pytest.raises(ValueError, match="x must contain exactly"):
        serializer.fit([graph])


def test_invalid_edge_feature_length_raises():
    serializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer()
    graph = SimpleGraph(
        edge_index=[[0, 1], [1, 2]], x=[7, 8, 9], edge_attr=[3])

    with pytest.raises(ValueError, match="edge_attr must contain exactly"):
        serializer.fit([graph])


def test_missing_node_features_supported():
    serializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer()
    graph = SimpleGraph(
        edge_index=[[0], [1]], x=None, edge_attr=[3], num_nodes=2)

    result = serializer.fit([graph]).serialize(graph)

    assert result.token_ids == [
        serializer.node_token(0), serializer.node_reference_token(0),
        serializer.edge_token(3),
        serializer.node_token(1), serializer.node_reference_token(1),
        serializer.edge_token(3),
        serializer.node_token(0), serializer.node_reference_token(0),
    ]


def test_missing_edge_features_supported():
    serializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer()
    graph = SimpleGraph(
        edge_index=[[0], [1]], x=[7, 8], edge_attr=None)

    result = serializer.fit([graph]).serialize(graph)

    assert result.token_ids == [
        serializer.node_token(7), serializer.node_reference_token(0),
        serializer.edge_token(0),
        serializer.node_token(8), serializer.node_reference_token(1),
        serializer.edge_token(0),
        serializer.node_token(7), serializer.node_reference_token(0),
    ]


def test_feuler_serialization_prefers_high_frequency_edge():
    FrequencyGuidedEulerianSerializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer

    training_graphs = [
        SimpleGraph(edge_index=[[0], [1]], x=[1, 2], edge_attr=[5]),
        SimpleGraph(edge_index=[[0], [1]], x=[1, 2], edge_attr=[5]),
        SimpleGraph(edge_index=[[0], [1]], x=[1, 3], edge_attr=[9]),
    ]
    target = SimpleGraph(edge_index=[[0, 0], [1, 2]], x=[1, 2, 3], edge_attr=[5, 9])

    serializer = FrequencyGuidedEulerianSerializer()
    serializer.fit(training_graphs)
    result = serializer.serialize(target)

    node_one = serializer.node_token(1)
    node_two = serializer.node_token(2)
    edge_five = serializer.edge_token(5)

    assert result.token_ids[:5] == [
        node_one,
        serializer.node_reference_token(0),
        edge_five,
        node_two,
        serializer.node_reference_token(1),
    ]
    assert result.metadata["method"] == "feuler"
    assert result.metadata["num_edges_traversed"] == 4


def test_feuler_serialization_sorts_and_joins_components():
    FrequencyGuidedEulerianSerializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer

    graph = SimpleGraph(edge_index=[[0, 2], [1, 3]], x=[4, 5, 6, 7], edge_attr=[1, 2])
    serializer = FrequencyGuidedEulerianSerializer(component_sep_token_id=-1)

    result = serializer.fit([graph]).serialize(graph)

    assert -1 in result.token_ids
    assert result.metadata["num_components"] == 2


def test_feuler_serializes_long_graph_without_python_recursion():
    FrequencyGuidedEulerianSerializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer
    num_edges = 1200
    graph = SimpleGraph(
        edge_index=[list(range(num_edges)), list(range(1, num_edges + 1))],
        x=[index % 5 for index in range(num_edges + 1)],
        edge_attr=[1] * num_edges,
    )

    result = FrequencyGuidedEulerianSerializer().fit([graph]).serialize(graph)

    assert result.metadata["num_edges_traversed"] == num_edges * 2
    assert len(result.token_ids) == num_edges * 6 + 2


def test_feuler_deserializes_from_token_stream_without_graph_metadata():
    FrequencyGuidedEulerianSerializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer

    graph = SimpleGraph(edge_index=[[0, 1], [1, 2]], x=[4, 5, 6], edge_attr=[1, 2])
    serializer = FrequencyGuidedEulerianSerializer().fit([graph])

    result = serializer.serialize(graph)
    restored = serializer.deserialize(result)

    assert restored["edge_index"] == [[0, 1], [1, 2]]
    assert restored["x"] == [4, 5, 6]
    assert restored["edge_attr"] == [1, 2]
    assert "node_labels" not in result.metadata
    assert "input_edges" not in result.metadata


def _canonical_labeled_edges(edge_index, edge_attr):
    labels = edge_attr if edge_attr is not None else [0] * len(edge_index[0])
    return sorted(
        (min(src, dst), max(src, dst), label)
        for src, dst, label in zip(edge_index[0], edge_index[1], labels))


def _component_sets(num_nodes, edge_index):
    neighbors = {node: set() for node in range(num_nodes)}
    for src, dst in zip(edge_index[0], edge_index[1]):
        neighbors[src].add(dst)
        neighbors[dst].add(src)
    components = []
    remaining = set(range(num_nodes))
    while remaining:
        pending = [min(remaining)]
        component = set()
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(neighbors[node] - component)
        remaining -= component
        components.append(frozenset(component))
    return set(components)


def _assert_feuler_round_trip(graph):
    serializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer().fit([graph])
    reconstructed = serializer.deserialize(serializer.serialize(graph))
    input_x = graph.x if graph.x is not None else list(range(graph.num_nodes))

    assert reconstructed["num_nodes"] == graph.num_nodes
    assert reconstructed["x"] == input_x
    assert _canonical_labeled_edges(
        reconstructed["edge_index"], reconstructed["edge_attr"]) == _canonical_labeled_edges(
            graph.edge_index, graph.edge_attr)
    assert _component_sets(
        reconstructed["num_nodes"], reconstructed["edge_index"]) == _component_sets(
            graph.num_nodes, graph.edge_index)


def test_round_trip_single_edge():
    _assert_feuler_round_trip(SimpleGraph(
        edge_index=[[0], [1]], x=[4, 9], edge_attr=[7]))


def test_round_trip_path():
    _assert_feuler_round_trip(SimpleGraph(
        edge_index=[[0, 1], [1, 2]], x=[4, 9, 4], edge_attr=[7, 8]))


def test_round_trip_cycle():
    _assert_feuler_round_trip(SimpleGraph(
        edge_index=[[0, 1, 2], [1, 2, 0]], x=[1, 1, 1], edge_attr=[2, 3, 4]))


def test_round_trip_star():
    _assert_feuler_round_trip(SimpleGraph(
        edge_index=[[0, 0, 0], [1, 2, 3]], x=[5, 5, 5, 5], edge_attr=[1, 2, 3]))


def test_round_trip_disconnected_graph():
    _assert_feuler_round_trip(SimpleGraph(
        edge_index=[[0, 2], [1, 3]], x=[1, 2, 3, 4], edge_attr=[5, 6]))


def test_round_trip_isolated_nodes():
    _assert_feuler_round_trip(SimpleGraph(
        edge_index=[[0], [1]], x=[1, 2, 3, 4], edge_attr=[5], num_nodes=4))


def test_round_trip_node_labels():
    _assert_feuler_round_trip(SimpleGraph(
        edge_index=[[0, 1], [1, 2]], x=[-3, 0, 11], edge_attr=[5, 5]))


def test_round_trip_edge_labels():
    _assert_feuler_round_trip(SimpleGraph(
        edge_index=[[0, 1], [1, 2]], x=[1, 2, 3], edge_attr=[-4, 9]))


def _feuler_structure_tokens(token_ids):
    """Remove reversible node references before comparing traversals."""
    return [token for token in token_ids if token == -1 or token % 3 != 2]


def _relabel_graph(graph, old_to_new):
    num_nodes = graph.num_nodes
    new_to_old = {new: old for old, new in old_to_new.items()}
    return SimpleGraph(
        edge_index=[
            [old_to_new[node] for node in graph.edge_index[0]],
            [old_to_new[node] for node in graph.edge_index[1]],
        ],
        x=[graph.x[new_to_old[node]] for node in range(num_nodes)],
        edge_attr=list(graph.edge_attr),
        num_nodes=num_nodes,
    )


def test_feuler_structure_tokens_are_stable_under_node_relabeling():
    FrequencyGuidedEulerianSerializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer
    graph = SimpleGraph(
        edge_index=[[0, 1, 1, 3], [1, 2, 3, 4]],
        x=[9, 1, 4, 7, 4], edge_attr=[3, 2, 5, 6], num_nodes=5)
    relabeled = _relabel_graph(graph, {0: 4, 1: 2, 2: 0, 3: 3, 4: 1})
    serializer = FrequencyGuidedEulerianSerializer().fit([graph])

    original = serializer.serialize(graph)
    permuted = serializer.serialize(relabeled)

    assert _feuler_structure_tokens(original.token_ids) == _feuler_structure_tokens(
        permuted.token_ids)


def test_feuler_frequency_is_stable_under_node_relabeling():
    FrequencyGuidedEulerianSerializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer
    graph = SimpleGraph(
        edge_index=[[0, 1, 1], [1, 2, 3]],
        x=[9, 1, 4, 7], edge_attr=[3, 2, 5], num_nodes=4)
    relabeled = _relabel_graph(graph, {0: 3, 1: 2, 2: 0, 3: 1})

    assert FrequencyGuidedEulerianSerializer().fit([graph]).frequency_map == (
        FrequencyGuidedEulerianSerializer().fit([relabeled]).frequency_map)


@pytest.mark.parametrize(
    "graph, old_to_new",
    [
        (
            SimpleGraph(
                edge_index=[[0, 1, 2, 3], [1, 2, 3, 0]],
                x=[1, 1, 1, 1], edge_attr=[2, 2, 2, 2], num_nodes=4),
            {0: 2, 1: 0, 2: 3, 3: 1},
        ),
        (
            SimpleGraph(
                edge_index=[[0, 0, 0, 0], [1, 2, 3, 4]],
                x=[9, 1, 1, 1, 1], edge_attr=[2, 2, 2, 2], num_nodes=5),
            {0: 3, 1: 0, 2: 4, 3: 1, 4: 2},
        ),
        (
            SimpleGraph(
                edge_index=[
                    [0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
                    [1, 2, 3, 4, 2, 3, 4, 3, 4, 4],
                ],
                x=[1, 1, 1, 1, 1], edge_attr=[2] * 10, num_nodes=5),
            {0: 4, 1: 2, 2: 0, 3: 3, 4: 1},
        ),
    ],
    ids=["cycle", "star_with_equivalent_leaves", "regular_graph"],
)
def test_feuler_symmetric_graphs_compare_by_structural_equivalence(graph, old_to_new):
    FrequencyGuidedEulerianSerializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer
    relabeled = _relabel_graph(graph, old_to_new)
    serializer = FrequencyGuidedEulerianSerializer().fit([graph])

    original = serializer.serialize(graph)
    permuted = serializer.serialize(relabeled)

    assert _feuler_structure_tokens(original.token_ids) == _feuler_structure_tokens(
        permuted.token_ids)
    _assert_feuler_round_trip(graph)
    _assert_feuler_round_trip(relabeled)


def _random_node_permutations(num_nodes, seed=0, count=50):
    rng = random.Random(seed)
    for _ in range(count):
        node_ids = list(range(num_nodes))
        rng.shuffle(node_ids)
        yield dict(enumerate(node_ids))


def _assert_feuler_permutation_regression(graph, seed=0):
    FrequencyGuidedEulerianSerializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer
    serializer = FrequencyGuidedEulerianSerializer().fit([graph])
    expected = _feuler_structure_tokens(serializer.serialize(graph).token_ids)
    _assert_feuler_round_trip(graph)
    for old_to_new in _random_node_permutations(graph.num_nodes, seed=seed):
        permuted = _relabel_graph(graph, old_to_new)
        assert _feuler_structure_tokens(serializer.serialize(permuted).token_ids) == expected
        _assert_feuler_round_trip(permuted)


def test_feuler_path_permutation():
    _assert_feuler_permutation_regression(SimpleGraph(
        edge_index=[[0, 1, 2, 3], [1, 2, 3, 4]],
        x=[8, 1, 4, 7, 9], edge_attr=[3, 2, 5, 6], num_nodes=5), seed=11)


def test_feuler_tree_permutation():
    _assert_feuler_permutation_regression(SimpleGraph(
        edge_index=[[0, 0, 1, 1, 2], [1, 2, 3, 4, 5]],
        x=[9, 1, 4, 7, 3, 8], edge_attr=[1, 2, 3, 4, 5], num_nodes=6), seed=13)


def test_feuler_star_permutation():
    _assert_feuler_permutation_regression(SimpleGraph(
        edge_index=[[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]],
        x=[9, 1, 1, 1, 1, 1], edge_attr=[2, 2, 2, 2, 2], num_nodes=6), seed=17)


def test_feuler_cycle_permutation():
    _assert_feuler_permutation_regression(SimpleGraph(
        edge_index=[[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
        x=[1, 1, 1, 1, 1], edge_attr=[2, 2, 2, 2, 2], num_nodes=5), seed=19)


def test_feuler_disconnected_permutation():
    _assert_feuler_permutation_regression(SimpleGraph(
        edge_index=[[0, 1, 3, 3], [1, 2, 4, 5]],
        x=[8, 1, 4, 9, 2, 7], edge_attr=[3, 2, 5, 6], num_nodes=6), seed=23)


def test_feuler_frequency_tie_independent_of_node_ids():
    FrequencyGuidedEulerianSerializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer
    graph = SimpleGraph(
        edge_index=[[0, 0, 0], [1, 2, 3]],
        x=[0, 7, 8, 9], edge_attr=[1, 2, 3], num_nodes=4)
    serializer = FrequencyGuidedEulerianSerializer().fit([graph])
    graph_view = serializer._read_graph(graph)
    priorities = [serializer._arc_priority(graph_view, 0, dst, label)
                  for dst, label in zip([1, 2, 3], [1, 2, 3])]

    assert priorities == [1, 1, 1]
    _assert_feuler_permutation_regression(graph, seed=29)


def test_feuler_round_trip_after_random_permutation():
    _assert_feuler_permutation_regression(SimpleGraph(
        edge_index=[[0, 0, 1, 2, 4], [1, 2, 3, 4, 5]],
        x=[9, 1, 4, 7, 3, 8], edge_attr=[1, 2, 3, 4, 5], num_nodes=6), seed=31)


def test_round_trip_after_bpe_encode_decode():
    graph = SimpleGraph(
        edge_index=[[0, 1, 2], [1, 2, 0]], x=[3, 3, 3], edge_attr=[4, 4, 4])
    graph_tokenizer = _load_graph_tokenizer_module()
    graph_bpe = _load_graph_bpe_module()
    tokenizer = graph_tokenizer.GraphTokenizer(
        bpe=graph_bpe.GraphBPE(num_merges=2, min_frequency=2)).fit([graph, graph])

    reconstructed = tokenizer.decode_graph(tokenizer.encode_graph(graph))

    assert reconstructed["num_nodes"] == graph.num_nodes
    assert reconstructed["x"] == graph.x
    assert _canonical_labeled_edges(
        reconstructed["edge_index"], reconstructed["edge_attr"]) == _canonical_labeled_edges(
            graph.edge_index, graph.edge_attr)
    assert _component_sets(
        reconstructed["num_nodes"], reconstructed["edge_index"]) == _component_sets(
            graph.num_nodes, graph.edge_index)


def test_feuler_rejects_graphs_that_cannot_be_an_undirected_simple_graph():
    FrequencyGuidedEulerianSerializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer
    graph = SimpleGraph(edge_index=[[0, 1], [1, 0]], x=[4, 5], edge_attr=[1, 2])

    with pytest.raises(ValueError, match="undirected simple"):
        FrequencyGuidedEulerianSerializer().fit([graph])


def test_feuler_rejects_parallel_edges_instead_of_deduplicating_them():
    FrequencyGuidedEulerianSerializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer
    graph = SimpleGraph(edge_index=[[0, 0], [1, 1]], x=[4, 5], edge_attr=[1, 1])

    with pytest.raises(ValueError, match="Parallel edges"):
        FrequencyGuidedEulerianSerializer().fit([graph])


def test_feuler_rejects_self_loops_and_explicit_directed_graphs():
    FrequencyGuidedEulerianSerializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer
    self_loop = SimpleGraph(edge_index=[[0], [0]], x=[4], edge_attr=[1])
    directed = {
        "edge_index": [[0], [1]], "x": [4, 5], "edge_attr": [1],
        "directed": True,
    }

    with pytest.raises(ValueError, match="self-loops"):
        FrequencyGuidedEulerianSerializer().serialize(self_loop)
    with pytest.raises(ValueError, match="directed graph semantics"):
        FrequencyGuidedEulerianSerializer().serialize(directed)


def test_serializer_preserves_preencoded_paper_node_and_edge_tokens():
    graph_serializer = _load_graph_serializer_module()
    graph = SimpleGraph(edge_index=[[0], [1]], x=[13, 17], edge_attr=[2])

    result = graph_serializer.FrequencyGuidedEulerianSerializer().fit([graph]).serialize(graph)

    serializer = graph_serializer.FrequencyGuidedEulerianSerializer().fit([graph])
    assert result.token_ids == [
        serializer.node_token(13), serializer.node_reference_token(0),
        serializer.edge_token(2),
        serializer.node_token(17), serializer.node_reference_token(1),
        serializer.edge_token(2),
        serializer.node_token(13), serializer.node_reference_token(0),
    ]


def test_eulerian_serializer_does_not_collect_frequency_statistics():
    graph_serializer = _load_graph_serializer_module()
    graph = SimpleGraph(edge_index=[[0], [1]], x=[13, 17], edge_attr=[2])

    serializer = graph_serializer.EulerianSerializer().fit([graph])

    assert serializer.name == "eulerian"
    assert serializer.frequency_map == {}
    assert serializer.serialize(graph).metadata["method"] == "eulerian"


def test_graph_bpe_trains_most_frequent_pairs_and_encodes_sequences():
    GraphBPE = _load_graph_bpe_module().GraphBPE

    bpe = GraphBPE(num_merges=2, min_frequency=2)
    bpe.fit([[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 3], [4, 5]])

    assert bpe.codebook.merge_rules[0] == (1, 2, 6)
    assert bpe.codebook.merge_rules[1] == (6, 6, 7)
    assert bpe.codebook.vocab_size == 8
    assert bpe.encode([1, 2, 1, 2, 3]) == [7, 3]
    assert bpe.batch_encode([[1, 2], [4, 5]]) == [[6], [4, 5]]


def test_graph_bpe_never_merges_pairs_touching_protected_tokens():
    GraphBPE = _load_graph_bpe_module().GraphBPE

    bpe = GraphBPE(num_merges=2, min_frequency=2, protected_token_ids={7})
    bpe.fit([[10, 7, 11], [10, 7, 11]])

    assert bpe.codebook.merge_rules == []
    assert bpe.encode([10, 7, 11]) == [10, 7, 11]


def test_graph_bpe_cpp_trains_protected_segments_with_the_native_backend(monkeypatch):
    GraphBPE = _load_graph_bpe_module().GraphBPE
    from third_party import graph_bpe_cpp

    calls = []

    monkeypatch.setattr(graph_bpe_cpp, "is_available", lambda: True)
    monkeypatch.setattr(
        graph_bpe_cpp,
        "train_bpe",
        lambda sequences, num_merges, min_frequency, initial_vocab_size: calls.append(
            (sequences, num_merges, min_frequency, initial_vocab_size)) or {
                "merge_rules": [(1, 2, 8)],
                "vocab_size": 9,
                "metadata": {"backend": "cpp"},
            },
    )

    bpe = GraphBPE(
        num_merges=1,
        min_frequency=2,
        backend="cpp",
        protected_token_ids={7},
    ).fit([[1, 2, 7, 3, 4], [1, 2, 7, 3, 4]])

    assert calls == [([[1, 2], [3, 4], [1, 2], [3, 4]], 1, 2, 8)]
    assert bpe.codebook.metadata["backend"] == "cpp"


def test_graph_bpe_cpp_rejects_missing_native_backend_with_protected_tokens(monkeypatch):
    GraphBPE = _load_graph_bpe_module().GraphBPE
    from third_party import graph_bpe_cpp

    monkeypatch.setattr(graph_bpe_cpp, "is_available", lambda: False)

    with pytest.raises(ImportError, match="native extension"):
        GraphBPE(backend="cpp", protected_token_ids={7}).fit([[1, 7, 2]])


def test_graph_bpe_respects_min_frequency_and_roundtrips_codebook(tmp_path):
    graph_bpe = _load_graph_bpe_module()
    GraphBPE = graph_bpe.GraphBPE

    bpe = GraphBPE(num_merges=5, min_frequency=3).fit([[9, 10], [9, 10], [1, 2]])

    assert bpe.codebook.merge_rules == []
    assert bpe.encode([9, 10]) == [9, 10]

    trained = GraphBPE(num_merges=1, min_frequency=2).fit([[9, 10], [9, 10], [1, 2]])
    codebook_path = tmp_path / "codebook.json"
    trained.save_codebook(codebook_path)

    loaded = GraphBPE.load_codebook(codebook_path)
    assert loaded.codebook.merge_rules == [(9, 10, 11)]
    assert loaded.encode([9, 10, 9, 10]) == [11, 11]


def test_graph_bpe_auto_backend_matches_python_fallback():
    GraphBPE = _load_graph_bpe_module().GraphBPE

    sequences = [[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 3], [4, 5]]
    python_bpe = GraphBPE(num_merges=2, min_frequency=2, backend="python").fit(sequences)
    auto_bpe = GraphBPE(num_merges=2, min_frequency=2, backend="auto").fit(sequences)

    assert auto_bpe.codebook.merge_rules == python_bpe.codebook.merge_rules
    assert auto_bpe.codebook.vocab_size == python_bpe.codebook.vocab_size
    assert auto_bpe.codebook.metadata["backend"] in {"cpp", "python"}
    assert auto_bpe.encode([1, 2, 1, 2, 3]) == python_bpe.encode([1, 2, 1, 2, 3])


def test_graph_bpe_cpp_matches_python_with_protected_separators_when_available():
    GraphBPE = _load_graph_bpe_module().GraphBPE
    from third_party import graph_bpe_cpp

    if not graph_bpe_cpp.is_available():
        pytest.skip("graph_bpe_cpp native extension is unavailable")
    separator = 7
    sequences = [[1, 2, separator, 1, 2], [1, 2, separator, 1, 2], [1, 2]]
    python_bpe = GraphBPE(
        num_merges=2, min_frequency=2, backend="python",
        protected_token_ids={separator}).fit(sequences)
    cpp_bpe = GraphBPE(
        num_merges=2, min_frequency=2, backend="cpp",
        protected_token_ids={separator}).fit(sequences)

    assert cpp_bpe.codebook.merge_rules == python_bpe.codebook.merge_rules
    assert cpp_bpe.codebook.vocab_size == python_bpe.codebook.vocab_size
    assert cpp_bpe.encode(sequences[0]) == python_bpe.encode(sequences[0])
    assert separator in cpp_bpe.encode(sequences[0])


def test_graph_bpe_auto_falls_back_without_the_native_bridge(monkeypatch):
    GraphBPE = _load_graph_bpe_module().GraphBPE
    from third_party import graph_bpe_cpp

    monkeypatch.setattr(graph_bpe_cpp, "is_available", lambda: False)
    monkeypatch.setattr(
        graph_bpe_cpp,
        "train_bpe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("native bridge used")),
    )

    bpe = GraphBPE(num_merges=1, min_frequency=2, backend="auto").fit(
        [[1, 2], [1, 2]])

    assert bpe.codebook.merge_rules == [(1, 2, 3)]
    assert bpe.codebook.metadata["backend"] == "python"


def test_graph_bpe_cpp_requires_the_native_bridge(monkeypatch):
    GraphBPE = _load_graph_bpe_module().GraphBPE
    from third_party import graph_bpe_cpp

    monkeypatch.setattr(graph_bpe_cpp, "is_available", lambda: False)

    bpe = GraphBPE(num_merges=1, min_frequency=2, backend="cpp")
    with pytest.raises(ImportError, match="native extension"):
        bpe.fit([[1, 2], [1, 2]])

    bpe.codebook.merge_rules = [(1, 2, 3)]
    with pytest.raises(ImportError, match="native extension"):
        bpe.encode([1, 2])
    with pytest.raises(ImportError, match="native extension"):
        bpe.batch_encode([[1, 2]])


def test_graph_bpe_cpp_package_exposes_native_results_when_available():
    from third_party import graph_bpe_cpp

    if not graph_bpe_cpp.is_available():
        with pytest.raises(ImportError, match="native extension"):
            graph_bpe_cpp.train_bpe([[1, 2]], num_merges=1, min_frequency=2)
        return

    result = graph_bpe_cpp.train_bpe(
        [[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 3]],
        num_merges=2,
        min_frequency=2,
    )

    assert result["merge_rules"] == [(1, 2, 4), (4, 4, 5)]
    assert result["vocab_size"] == 6
    assert graph_bpe_cpp.encode([1, 2, 1, 2, 3], result["merge_rules"]) == [5, 3]


def test_graph_bpe_cpp_batch_encode_crosses_the_bridge_once(monkeypatch):
    graph_bpe = _load_graph_bpe_module()
    from third_party import graph_bpe_cpp

    calls = []

    def batch_encode(token_sequences, merge_rules):
        calls.append((token_sequences, merge_rules))
        return [[6], [4, 5]]

    def fail_single_encode(*_args, **_kwargs):
        raise AssertionError("batch encoding must not cross the native bridge once per sequence")

    monkeypatch.setattr(graph_bpe_cpp, "batch_encode", batch_encode, raising=False)
    monkeypatch.setattr(graph_bpe_cpp, "encode", fail_single_encode)
    monkeypatch.setattr(graph_bpe_cpp, "is_available", lambda: True)
    engine = graph_bpe.GraphBPE(num_merges=0, backend="cpp")
    engine.codebook = graph_bpe.BPECodebook(merge_rules=[(1, 2, 6)], vocab_size=7)

    encoded = engine.batch_encode([[1, 2], [4, 5]])

    assert encoded == [[6], [4, 5]]
    assert calls == [([[1, 2], [4, 5]], [(1, 2, 6)])]


def test_graph_bpe_auto_uses_native_batch_for_native_codebook(monkeypatch):
    graph_bpe = _load_graph_bpe_module()
    from third_party import graph_bpe_cpp

    calls = []
    monkeypatch.setattr(
        graph_bpe_cpp,
        "batch_encode",
        lambda sequences, rules: calls.append((sequences, rules)) or [[6]],
    )
    monkeypatch.setattr(graph_bpe_cpp, "is_available", lambda: True)
    engine = graph_bpe.GraphBPE(num_merges=0, backend="auto")
    engine.codebook = graph_bpe.BPECodebook(
        merge_rules=[(1, 2, 6)],
        vocab_size=7,
        metadata={"backend": "cpp"},
    )

    assert engine.batch_encode([[1, 2]]) == [[6]]
    assert calls == [([[1, 2]], [(1, 2, 6)])]


def test_graph_tokenizer_fits_serializer_and_bpe_then_encodes_graph():
    graph_tokenizer = _load_graph_tokenizer_module()
    graph_bpe = _load_graph_bpe_module()
    GraphTokenizer = graph_tokenizer.GraphTokenizer
    GraphBPE = graph_bpe.GraphBPE

    graph = SimpleGraph(edge_index=[[0], [1]], x=[1, 2], edge_attr=[5])
    tokenizer = GraphTokenizer(bpe=GraphBPE(num_merges=1, min_frequency=2))
    tokenizer.fit([graph, graph])

    encoding = tokenizer.encode_graph(graph)

    assert tokenizer.bpe.codebook.merge_rules
    assert encoding.input_ids[0] == tokenizer.special_tokens.cls_token_id
    assert encoding.input_ids[-1] == tokenizer.special_tokens.sep_token_id
    assert encoding.attention_mask == [1] * len(encoding.input_ids)
    assert encoding.metadata["serializer"]["method"] == "feuler"


def test_graph_tokenizer_protects_component_separator_from_bpe_merges():
    graph_tokenizer = _load_graph_tokenizer_module()
    graph_bpe = _load_graph_bpe_module()
    graph = SimpleGraph(
        edge_index=[[0, 2], [1, 3]], x=[1, 2, 1, 2], edge_attr=[5, 5])
    tokenizer = graph_tokenizer.GraphTokenizer(
        bpe=graph_bpe.GraphBPE(num_merges=8, min_frequency=2))

    tokenizer.fit([graph, graph])

    separator = tokenizer.special_tokens.component_sep_token_id
    assert all(separator not in rule[:2] for rule in tokenizer.bpe.codebook.merge_rules)


def test_graph_tokenizer_rejects_empty_training_corpus_and_unfitted_encoding():
    graph_tokenizer = _load_graph_tokenizer_module()
    graph_bpe = _load_graph_bpe_module()
    graph = SimpleGraph(edge_index=[[0], [1]], x=[1, 2], edge_attr=[5])
    tokenizer = graph_tokenizer.GraphTokenizer(bpe=graph_bpe.GraphBPE(num_merges=0))

    with pytest.raises(ValueError, match="empty training corpus"):
        tokenizer.fit([])
    with pytest.raises(RuntimeError, match="must be fit"):
        tokenizer.encode_graph(graph)


def test_graph_tokenizer_records_train_only_fit_provenance():
    graph_tokenizer = _load_graph_tokenizer_module()
    graph_bpe = _load_graph_bpe_module()
    graph = SimpleGraph(edge_index=[[0], [1]], x=[1, 2], edge_attr=[5])
    tokenizer = graph_tokenizer.GraphTokenizer(bpe=graph_bpe.GraphBPE(num_merges=0))

    tokenizer.fit([graph], graph_ids=[42])
    encoding = tokenizer.encode_graph(graph)

    assert tokenizer.fit_graph_ids_hash
    assert encoding.metadata["tokenizer_schema_version"] == 1
    assert encoding.metadata["fit_graph_ids_hash"] == tokenizer.fit_graph_ids_hash


def test_frozen_train_vocabulary_maps_unseen_validation_and_test_tokens_to_unk():
    graph_tokenizer = _load_graph_tokenizer_module()
    graph_bpe = _load_graph_bpe_module()
    train = SimpleGraph(edge_index=[[0], [1]], x=[1, 2], edge_attr=[5])
    held_out = SimpleGraph(edge_index=[[0], [1]], x=[101, 102], edge_attr=[5])
    tokenizer = graph_tokenizer.GraphTokenizer(
        bpe=graph_bpe.GraphBPE(num_merges=1, min_frequency=2)).fit([train, train])
    before = (dict(tokenizer.vocabulary), list(tokenizer.bpe.codebook.merge_rules))

    validation = tokenizer.encode_graph(held_out)
    test = tokenizer.encode_graph(held_out)

    assert (dict(tokenizer.vocabulary), list(tokenizer.bpe.codebook.merge_rules)) == before
    assert tokenizer.special_tokens.unk_token_id in validation.input_ids
    assert tokenizer.special_tokens.unk_token_id in test.input_ids
    assert all(0 <= token <= tokenizer.max_token_id for token in validation.input_ids)
    assert all(0 <= token <= tokenizer.max_token_id for token in test.input_ids)


def test_train_vocabulary_and_bpe_merge_ids_are_contiguous_and_unique():
    graph_tokenizer = _load_graph_tokenizer_module()
    graph_bpe = _load_graph_bpe_module()
    graph = SimpleGraph(edge_index=[[0], [1]], x=[1, 2], edge_attr=[5])
    tokenizer = graph_tokenizer.GraphTokenizer(
        bpe=graph_bpe.GraphBPE(num_merges=2, min_frequency=2)).fit([graph, graph])
    vocabulary_ids = sorted(tokenizer.vocabulary.values())
    merge_ids = [rule[2] for rule in tokenizer.bpe.codebook.merge_rules]

    assert vocabulary_ids == list(range(8, tokenizer.max_token_id + 1))
    assert len(merge_ids) == len(set(merge_ids))


def test_graph_tokenizer_batch_encodes_serialized_graphs_together(monkeypatch):
    graph_tokenizer = _load_graph_tokenizer_module()
    graph_bpe = _load_graph_bpe_module()
    graph_one = SimpleGraph(edge_index=[[0], [1]], x=[1, 2], edge_attr=[5])
    graph_two = SimpleGraph(edge_index=[[0], [1]], x=[1, 3], edge_attr=[6])
    tokenizer = graph_tokenizer.GraphTokenizer(
        bpe=graph_bpe.GraphBPE(num_merges=0)).fit([graph_one, graph_two])
    calls = []

    def batch_encode(sequences):
        calls.append(sequences)
        return [list(sequence) for sequence in sequences]

    monkeypatch.setattr(tokenizer.bpe, "batch_encode", batch_encode)

    results = tokenizer.batch_encode_graphs([graph_one, graph_two])

    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert [result.input_ids[0] for result in results] == [
        tokenizer.special_tokens.cls_token_id,
        tokenizer.special_tokens.cls_token_id,
    ]


def test_graph_tokenizer_call_attaches_token_fields_to_graph():
    graph_tokenizer = _load_graph_tokenizer_module()
    graph_bpe = _load_graph_bpe_module()
    GraphTokenizer = graph_tokenizer.GraphTokenizer
    GraphBPE = graph_bpe.GraphBPE

    graph = SimpleGraph(edge_index=[[0], [1]], x=[1, 2], edge_attr=[5])
    tokenizer = GraphTokenizer(bpe=GraphBPE(num_merges=1, min_frequency=2)).fit([graph, graph])

    transformed = tokenizer(graph)

    assert transformed is graph
    assert graph.input_ids == tokenizer.encode_graph(graph).input_ids
    assert graph.attention_mask == [1] * len(graph.input_ids)
    assert graph.graph_tokenizer_metadata["serializer"]["method"] == "feuler"


def test_graph_tokenizer_builds_mlm_pretraining_batch():
    graph_tokenizer = _load_graph_tokenizer_module()
    graph_bpe = _load_graph_bpe_module()
    GraphTokenizer = graph_tokenizer.GraphTokenizer
    GraphBPE = graph_bpe.GraphBPE

    graph = SimpleGraph(edge_index=[[0], [1]], x=[1, 2], edge_attr=[5])
    tokenizer = GraphTokenizer(bpe=GraphBPE(num_merges=0, min_frequency=2)).fit([graph])

    batch = tokenizer.build_mlm_batch([graph], max_length=10, mask_prob=1.0, seed=3)
    sequence = batch.input_ids[0]
    labels = batch.labels[0]

    assert len(sequence) == 10
    assert len(batch.attention_mask[0]) == 10
    assert sequence[0] == tokenizer.special_tokens.cls_token_id
    assert labels[0] == -100
    assert tokenizer.special_tokens.mask_token_id in sequence
    assert sum(1 for value in labels if value != -100) > 0
    assert batch.metadata["num_mlm_labels"] == sum(1 for value in labels if value != -100)


def test_graph_tokenizer_builds_mlm_batch_from_token_sequences():
    tokenizer = _load_graph_tokenizer_module().GraphTokenizer()

    batch = tokenizer.build_mlm_batch_from_token_sequences(
        [[3, 6, 4], [3, 8, 9, 4]], max_length=5, mask_prob=1.0, seed=7)

    assert batch.input_ids == [[3, 2, 4, 0, 0], [3, 2, 2, 4, 0]]
    assert batch.attention_mask == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]
    assert batch.labels == [[-100, 6, -100, -100, -100], [-100, 8, 9, -100, -100]]


def test_graph_tokenizer_rejects_sequence_overflow_instead_of_truncating():
    tokenizer = _load_graph_tokenizer_module().GraphTokenizer()

    with pytest.raises(ValueError, match="exceeds max_length"):
        tokenizer.build_mlm_batch_from_token_sequences(
            [[3, 6, 7, 4]], max_length=3, mask_prob=0.0)


def test_serializer_rejects_multidimensional_features_without_an_adapter():
    serializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer()
    graph = SimpleGraph(edge_index=[[0], [1]], x=[[1, 2], [1, 9]], edge_attr=[[5, 6]])

    with pytest.raises(ValueError, match="Multi-dimensional"):
        serializer.fit([graph])


def test_graph_tokenizer_mlm_forces_one_mask_when_random_draw_selects_none():
    graph_tokenizer = _load_graph_tokenizer_module()
    tokenizer = graph_tokenizer.GraphTokenizer()
    input_ids = [[
        tokenizer.special_tokens.cls_token_id,
        11,
        12,
        tokenizer.special_tokens.sep_token_id,
    ]]

    masked, labels = tokenizer._mask_input_ids(
        input_ids, mask_prob=1e-12, seed=0)

    selected = [
        (row, column)
        for row, label_row in enumerate(labels)
        for column, label in enumerate(label_row)
        if label != -100
    ]
    assert len(selected) == 1
    row, column = selected[0]
    assert labels[row][column] == input_ids[row][column]
    assert masked[row][column] == tokenizer.special_tokens.mask_token_id


def test_graph_tokenizer_replaces_component_separator_before_mlm():
    graph_tokenizer = _load_graph_tokenizer_module()
    graph_bpe = _load_graph_bpe_module()
    GraphTokenizer = graph_tokenizer.GraphTokenizer
    GraphBPE = graph_bpe.GraphBPE

    graph = SimpleGraph(edge_index=[[0, 2], [1, 3]], x=[1, 2, 3, 4], edge_attr=[5, 6])
    tokenizer = GraphTokenizer(bpe=GraphBPE(num_merges=0, min_frequency=2)).fit([graph])

    encoding = tokenizer.encode_graph(graph)
    batch = tokenizer.build_mlm_batch([graph], max_length=20, mask_prob=1.0, seed=4)
    sep_index = encoding.input_ids.index(tokenizer.special_tokens.component_sep_token_id)

    assert -1 not in encoding.input_ids
    assert tokenizer.special_tokens.component_sep_token_id in encoding.input_ids
    assert batch.input_ids[0][sep_index] == tokenizer.special_tokens.component_sep_token_id
    assert batch.labels[0][sep_index] == -100


def test_graph_tokenizer_rejects_a_model_vocabulary_smaller_than_encoded_tokens():
    graph_tokenizer = _load_graph_tokenizer_module()
    graph_bpe = _load_graph_bpe_module()
    GraphTokenizer = graph_tokenizer.GraphTokenizer
    GraphBPE = graph_bpe.GraphBPE

    graph = SimpleGraph(edge_index=[[0], [1]], x=[1, 2], edge_attr=[5])
    tokenizer = GraphTokenizer(bpe=GraphBPE(num_merges=1, min_frequency=2)).fit([graph, graph])

    with pytest.raises(ValueError, match="vocab_size"):
        tokenizer.validate_model_vocab(2)
