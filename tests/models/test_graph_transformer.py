import pytest
import tensorlayerx as tlx

from gammagl.models import GraphBERT, GraphGTE


@pytest.mark.parametrize("model_class", [GraphBERT, GraphGTE])
def test_graph_transformer_runs_supervised_and_mlm_forward(model_class):
    model = model_class(
        vocab_size=16,
        output_dim=2,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
        dropout_rate=0.0,
    )
    input_ids = tlx.convert_to_tensor([[1, 2, 3]], dtype=tlx.int64)
    attention_mask = tlx.convert_to_tensor([[1, 1, 1]], dtype=tlx.int64)

    supervised = model(input_ids, attention_mask=attention_mask, task="supervised")
    mlm = model(input_ids, attention_mask=attention_mask, task="mlm")

    assert tlx.get_tensor_shape(supervised) == [1, 2]
    assert tlx.get_tensor_shape(mlm) == [1, 3, 16]
