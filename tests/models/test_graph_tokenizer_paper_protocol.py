import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


def _load_paper_protocol_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "graph_tokenizer"
        / "paper_protocol.py"
    )
    spec = importlib.util.spec_from_file_location(
        "paper_protocol_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_protocol_loads_graph_transformer_classes():
    GraphBERT, GraphGTE = _load_paper_protocol_module()._load_paper_model_classes()

    assert GraphBERT.__name__ == "GraphBERT"
    assert GraphGTE.__name__ == "GraphGTE"


@pytest.mark.parametrize("encoder_type", ["bert", "gte"])
def test_paper_protocol_creates_a_model_and_runs_forward(encoder_type):
    protocol = _load_paper_protocol_module()
    model = protocol._create_paper_model(
        encoder_type=encoder_type,
        vocab_size=16,
        pad_token_id=0,
        task_type="regression",
        output_dim=2,
        strict_architecture=False,
        allow_random_gte_init=encoder_type == "gte",
        model_config={
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "max_position_embeddings": 16,
            "dropout_rate": 0.0,
        },
    )
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)

    supervised = model(input_ids, attention_mask, task="supervised")
    mlm = model(input_ids, attention_mask, task="mlm")

    assert supervised.shape == (1, 2)
    assert mlm.shape == (1, 3, 16)
