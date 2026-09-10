from gammagl.transforms import (
    FrequencyGuidedEulerianSerializer,
    GraphBPE,
    GraphTokenizer,
)


GRAPH = {
    "edge_index": [[0, 1], [1, 2]],
    "x": [6, 8, 7],
    "edge_attr": [1, 2],
    "num_nodes": 3,
}


def test_feuler_serializer_serializes_a_graph():
    serializer = FrequencyGuidedEulerianSerializer().fit([GRAPH])

    result = serializer.serialize(GRAPH)

    assert result.token_ids
    assert result.metadata["num_nodes"] == 3


def test_graph_bpe_encodes_and_decodes_token_sequence():
    bpe = GraphBPE(num_merges=1, min_frequency=2).fit([[1, 2, 1, 2]])

    encoded = bpe.encode([1, 2, 1, 2])

    assert bpe.codebook.merge_rules
    assert bpe.decode(encoded) == [1, 2, 1, 2]


def test_graph_tokenizer_fits_and_encodes_a_graph():
    tokenizer = GraphTokenizer(
        bpe=GraphBPE(num_merges=2, min_frequency=1)).fit([GRAPH])

    encoding = tokenizer.encode_graph(GRAPH)

    assert encoding.input_ids[0] == tokenizer.special_tokens.cls_token_id
    assert encoding.input_ids[-1] == tokenizer.special_tokens.sep_token_id
    assert len(encoding.input_ids) == len(encoding.attention_mask)
    assert encoding.serialized_token_ids
