import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorlayerx as tlx


if tlx.BACKEND != "torch":
    pytest.skip("Native torch metadata checks require TLX torch", allow_module_level=True)
torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def trainer():
    spec = importlib.util.spec_from_file_location(
        "graph_tokenizer_dataset_metadata_trainer",
        ROOT / "examples" / "graph_tokenizer" / "graph_tokenizer_trainer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("name", "canonical"), (
    ("mutagenicity", "mutagenicity"), ("mutag", "mutagenicity"), ("muta", "mutagenicity"),
    ("dblp", "dblp"), ("dblp-v1", "dblp"), ("dblp_v1", "dblp"),
    ("aqsol", "aqsol"),
))
def test_dataset_specs_resolve_canonical_names_and_aliases(trainer, name, canonical):
    assert trainer.resolve_dataset(name).name == canonical


def test_dataset_specs_expose_task_metadata_and_invalid_names_fail_fast(trainer):
    mutagenicity = trainer.resolve_dataset("mutagenicity")
    dblp = trainer.resolve_dataset("dblp")
    aqsol = trainer.resolve_dataset("aqsol")

    assert (mutagenicity.name, mutagenicity.task_type, mutagenicity.num_classes,
            mutagenicity.metric, mutagenicity.higher_is_better) == (
                "mutagenicity", "binary_classification", 2, "Accuracy", True)
    assert (dblp.name, dblp.dataset_class, dblp.storage_name) == (
        "dblp", "GraphTokenizerDBLP", "dblp")
    assert (aqsol.label_width, aqsol.num_classes, aqsol.metric,
            aqsol.higher_is_better) == (1, 0, "MAE", False)
    with pytest.raises(ValueError, match="mutagenicity.*dblp.*aqsol"):
        trainer.resolve_dataset("not-a-graph-tokenizer-dataset")


@pytest.mark.parametrize("dataset", ("mutagenicity", "dblp"))
def test_classification_synthetic_labels_and_normalization_are_task_driven(trainer, dataset):
    spec = trainer.resolve_dataset(dataset)
    splits = trainer.synthetic_splits(spec)
    encoded = {
        split: [{"labels": list(graph.y)} for graph in graphs]
        for split, graphs in splits.items()
    }

    normalizer, output_dim = trainer.normalize_labels(encoded, spec)

    assert {int(record["labels"][0]) for record in encoded["train"]} == {0, 1}
    assert normalizer is None
    assert output_dim == 2
    logits = torch.tensor([[3.0, 1.0], [0.5, 2.0]])
    labels = torch.tensor([[0.0], [1.0]])
    assert float(trainer.supervised_loss(torch, logits, labels, spec)) > 0.0


def test_aqsol_normalization_uses_only_training_labels_and_mse(trainer):
    spec = trainer.resolve_dataset("aqsol")
    encoded = {
        "train": [{"labels": [1.0], "graph_id": 0}, {"labels": [3.0], "graph_id": 1}],
        "val": [{"labels": [101.0], "graph_id": 0}],
        "test": [{"labels": [-99.0], "graph_id": 0}],
    }

    normalizer, output_dim = trainer.normalize_labels(encoded, spec)

    assert normalizer == {"mean": [2.0], "std": [1.0], "targets": ["solubility"]}
    assert output_dim == 1
    assert encoded["val"][0]["labels"] == [99.0]
    loss = trainer.supervised_loss(
        torch, torch.tensor([[2.0]]), torch.tensor([[1.0]]), spec)
    assert float(loss) == pytest.approx(1.0)


@pytest.mark.parametrize("extra_variants,val_label", [(1, 100.0), (4, -1000.0)])
def test_aqsol_normalization_counts_each_training_graph_once(trainer, extra_variants, val_label):
    encoded = {
        "train": ([{"graph_id": 0, "labels": [0.0]}] +
                  [{"graph_id": 1, "labels": [10.0]} for _ in range(extra_variants)]),
        "val": [{"graph_id": 0, "labels": [val_label]}],
        "test": [{"graph_id": 0, "labels": [200.0]}],
    }
    normalizer, _ = trainer.normalize_labels(encoded, trainer.resolve_dataset("aqsol"))
    assert normalizer["mean"] == [5.0]
    assert normalizer["std"] == [5.0]
    assert [record["labels"] for record in encoded["train"]] == (
        [[-1.0]] + [[1.0]] * extra_variants)
    assert encoded["val"][0]["labels"] == [(val_label - 5.0) / 5.0]
    assert encoded["test"][0]["labels"] == [39.0]


def test_aqsol_normalization_rejects_inconsistent_variants(trainer):
    encoded = {"train": [{"graph_id": 7, "labels": [0.0]},
                         {"graph_id": 7, "labels": [1.0]}],
               "val": [], "test": []}
    with pytest.raises(ValueError, match="graph_id 7.*inconsistent"):
        trainer.normalize_labels(encoded, trainer.resolve_dataset("aqsol"))


def test_aqsol_normalization_preserves_zero_variance_guard(trainer):
    encoded = {"train": [{"graph_id": 0, "labels": [4.0]},
                         {"graph_id": 1, "labels": [4.0]}],
               "val": [{"graph_id": 0, "labels": [5.0]}], "test": []}
    normalizer, _ = trainer.normalize_labels(encoded, trainer.resolve_dataset("aqsol"))
    assert normalizer["std"] == [1e-12]
    assert encoded["val"][0]["labels"] == [1e12]


def test_evaluation_uses_argmax_accuracy_and_raw_scale_mae(trainer):
    class FixedLogits(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.register_buffer("logits", torch.as_tensor(logits, dtype=torch.float32))

        def forward(self, input_ids, attention=None, task=None):
            return self.logits[:len(input_ids)]

    classification_loader = [(
        torch.ones((2, 3), dtype=torch.long), torch.ones((2, 3), dtype=torch.long),
        torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1]),
    )]
    classification = trainer.evaluate_downstream(
        torch, FixedLogits([[3.0, 1.0], [0.5, 2.0]]), classification_loader,
        trainer.resolve_dataset("mutagenicity"), None, torch.device("cpu"))
    assert classification["accuracy"] == pytest.approx(1.0)

    regression_loader = [(
        torch.ones((1, 3), dtype=torch.long), torch.ones((1, 3), dtype=torch.long),
        torch.tensor([[0.0]]), torch.tensor([0]),
    )]
    regression = trainer.evaluate_downstream(
        torch, FixedLogits([[1.0]]), regression_loader,
        trainer.resolve_dataset("aqsol"),
        {"mean": [2.0], "std": [3.0], "targets": ["solubility"]}, torch.device("cpu"))
    assert regression["mae"] == pytest.approx(3.0)


def test_paper_configs_cover_only_the_new_benchmarks(trainer):
    assert set(trainer.PAPER_CONFIGS) == {
        ("mutagenicity", "bert"), ("mutagenicity", "gte"),
        ("dblp", "bert"), ("dblp", "gte"),
        ("aqsol", "bert"), ("aqsol", "gte"),
    }
