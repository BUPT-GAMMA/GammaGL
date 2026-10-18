import numpy as np
import pytest
import tensorlayerx as tlx

from gammagl.data import Graph
from gammagl.transforms.graph_serializer import FrequencyGuidedEulerianSerializer
from gammagl.transforms.graph_tokenizer import GraphTokenizer


@pytest.mark.parametrize("representation", ["python", "numpy", "tensor"])
def test_graph_tokenizer_round_trips_backend_labels(representation):
    expected = {
        "edge_index": [[0, 1], [1, 2]],
        "x": [7, -2, 4],
        "edge_attr": [3, -1],
        "num_nodes": 3,
    }
    values = {"edge_index": [[0, 1], [1, 2]],
              "x": [[7], [-2], [4]], "edge_attr": [[3], [-1]]}
    if representation == "numpy":
        values = {name: np.asarray(value, dtype=np.int64)
                  for name, value in values.items()}
    elif representation == "tensor":
        values = {name: tlx.convert_to_tensor(value, dtype=tlx.int64)
                  for name, value in values.items()}
    graph = Graph(num_nodes=3)
    for name, value in values.items():
        setattr(graph, name, value)
    serializer = FrequencyGuidedEulerianSerializer(protocol="local_refs").fit([graph])

    assert serializer.restore_input_metadata(serializer.serialize(graph)) == expected

    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(protocol="local_refs")).fit([graph])
    encoding = tokenizer.encode_graph(graph)
    assert tokenizer.decode_graph(encoding) == expected
    assert encoding.attention_mask == [1] * len(encoding.input_ids)
    assert tokenizer.special_tokens.unk_token_id not in encoding.input_ids


@pytest.mark.parametrize("scalar", [int, np.int64, np.asarray,
                                    tlx.convert_to_tensor])
def test_serializer_accepts_zero_dimensional_integer_labels(scalar):
    graph = {"edge_index": [[0], [1]], "num_nodes": 2,
             "x": [scalar(7), scalar(-2)], "edge_attr": [scalar(3)]}
    serializer = FrequencyGuidedEulerianSerializer(protocol="local_refs").fit([graph])

    assert serializer.restore_input_metadata(serializer.serialize(graph)) == {
        "edge_index": [[0], [1]], "num_nodes": 2,
        "x": [7, -2], "edge_attr": [3],
    }


@pytest.mark.parametrize("value", [1.0, 1.25, True])
@pytest.mark.parametrize("convert", [lambda value: value, np.asarray,
                                     tlx.convert_to_tensor])
@pytest.mark.parametrize("label_field", ["x", "edge_attr"])
def test_serializer_rejects_noninteger_scalar_labels(value, convert, label_field):
    graph = {"edge_index": [[0], [1]], "num_nodes": 2,
             "x": [7, -2], "edge_attr": [3]}
    graph[label_field][0] = convert(value)

    with pytest.raises(ValueError, match="labels must be integers"):
        FrequencyGuidedEulerianSerializer(protocol="local_refs").fit([graph])
