import pytest


def test_graph_bpe_cpp_extension_trains_a_token_sequence():
    native = pytest.importorskip("third_party.graph_bpe_cpp._graph_bpe")

    result = native.train_bpe([[1, 2, 1, 2]], 1, 2)

    assert result["merge_rules"]
    assert result["vocab_size"] >= 3
