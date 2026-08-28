import importlib.util
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

    assert serializer.frequency_map[(7, 3, 8)] == 1
    assert serializer.frequency_map[(8, 3, 7)] == 1
    assert serializer.frequency_map[(8, 4, 9)] == 1


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

    assert result.token_ids[:3] == [node_one, edge_five, node_two]
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
    assert len(result.token_ids) == num_edges * 4 + 1


def test_feuler_only_restores_input_graph_from_serialization_metadata():
    FrequencyGuidedEulerianSerializer = _load_graph_serializer_module().FrequencyGuidedEulerianSerializer

    graph = SimpleGraph(edge_index=[[0, 1], [1, 2]], x=[4, 5, 6], edge_attr=[1, 2])
    serializer = FrequencyGuidedEulerianSerializer().fit([graph])

    result = serializer.serialize(graph)
    restored = serializer.restore_input_metadata(result)

    assert restored["edge_index"] == [[0, 1], [1, 2]]
    assert restored["x"] == [4, 5, 6]
    assert restored["edge_attr"] == [1, 2]
    with pytest.raises(NotImplementedError, match="not self-describing"):
        serializer.deserialize(result)


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


def test_serializer_preserves_preencoded_paper_node_and_edge_tokens():
    graph_serializer = _load_graph_serializer_module()
    graph = SimpleGraph(edge_index=[[0], [1]], x=[13, 17], edge_attr=[2])

    result = graph_serializer.FrequencyGuidedEulerianSerializer().fit([graph]).serialize(graph)

    assert result.token_ids == [13, 2, 17, 2, 13]


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

    batch = tokenizer.build_mlm_batch([graph], max_length=8, mask_prob=1.0, seed=3)
    sequence = batch.input_ids[0]
    labels = batch.labels[0]

    assert len(sequence) == 8
    assert len(batch.attention_mask[0]) == 8
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
    batch = tokenizer.build_mlm_batch([graph], max_length=16, mask_prob=1.0, seed=4)
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
