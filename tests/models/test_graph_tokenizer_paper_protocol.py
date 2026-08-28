import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _load_paper_protocol_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "graph_tokenizer"
        / "paper_protocol.py"
    )
    spec = importlib.util.spec_from_file_location("paper_protocol_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QM9_SPEC = SimpleNamespace(
    canonical_name="qm9",
    task_type="regression",
    output_dim=16,
    label_keys=(
        "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0",
        "u298", "h298", "g298", "cv", "u0_atom", "u298_atom",
        "h298_atom", "g298_atom",
    ),
)


def test_paper_protocol_loads_native_tlx_model_classes():
    protocol = _load_paper_protocol_module()

    GraphBERT, GraphGTE = protocol._load_paper_model_classes()

    assert GraphBERT.__name__ == "GraphBERT"
    assert GraphGTE.__name__ == "GraphGTE"
    assert Path(sys.modules[GraphBERT.__module__].__file__).resolve().name == "graph_bert.py"
    assert Path(sys.modules[GraphGTE.__module__].__file__).resolve().name == "graph_gte.py"


def test_paper_runtime_rejects_non_torch_tlx_backend(monkeypatch):
    protocol = _load_paper_protocol_module()
    monkeypatch.setenv("TL_BACKEND", "tensorflow")

    with pytest.raises(RuntimeError, match="TL_BACKEND=torch"):
        protocol.require_paper_runtime(require_cuda=False)


@pytest.mark.parametrize("encoder_type", ["bert", "gte"])
def test_paper_protocol_constructs_native_tlx_models(encoder_type):
    protocol = _load_paper_protocol_module()
    model = protocol._create_paper_model(
        encoder_type=encoder_type,
        vocab_size=24,
        pad_token_id=0,
        task_type="regression",
        output_dim=2,
        pooling="mean",
        strict_architecture=False,
        model_config={
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "max_position_embeddings": 16,
            "dropout_rate": 0.0,
        },
    )
    input_ids = torch.tensor([[3, 5, 4, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)

    assert model(input_ids, attention_mask, task="mlm").shape == (1, 4, 24)
    assert model(input_ids, attention_mask, task="supervised").shape == (1, 2)
    assert model.manifest()["encoder_type"] == encoder_type


def test_paper_protocol_rejects_non_fp32_model_parameters():
    protocol = _load_paper_protocol_module()
    model = torch.nn.Linear(4, 2).half()

    with pytest.raises(RuntimeError, match="FP32 model parameters"):
        protocol._require_fp32_parameters(torch, model)


def test_paper_protocol_accepts_qm9_multi_target_without_target_property():
    protocol = _load_paper_protocol_module()
    args = SimpleNamespace(dataset="qm9", target_property=None, runs=5, seed=42)

    protocol.validate_paper_args(args, QM9_SPEC)


def test_paper_protocol_rejects_legacy_qm9_single_target_selection():
    protocol = _load_paper_protocol_module()
    args = SimpleNamespace(dataset="qm9", target_property="homo", runs=5, seed=42)

    with pytest.raises(ValueError, match="16-target"):
        protocol.validate_paper_args(args, QM9_SPEC)


def test_paper_runtime_versions_accept_exact_paper_stack():
    protocol = _load_paper_protocol_module()
    versions = {
        "torch": "2.1.2+cu121",
        "cuda": "12.1",
        "dgl": "2.4.0+cu121",
        "torch_geometric": "2.4.0",
    }

    protocol.validate_paper_runtime_versions(versions)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("torch", "2.11.0+cu128"),
        ("cuda", "12.8"),
        ("dgl", "1.1.3"),
        ("torch_geometric", "2.8.0"),
    ],
)
def test_paper_runtime_versions_reject_nonpaper_stack(field, wrong_value):
    protocol = _load_paper_protocol_module()
    versions = {
        "torch": "2.1.2+cu121",
        "cuda": "12.1",
        "dgl": "2.4.0+cu121",
        "torch_geometric": "2.4.0",
    }
    versions[field] = wrong_value

    with pytest.raises(RuntimeError, match=field):
        protocol.validate_paper_runtime_versions(versions)


def test_qm9_prepare_labels_normalizes_every_target_from_train_only():
    protocol = _load_paper_protocol_module()
    spec = SimpleNamespace(
        canonical_name="qm9",
        output_dim=2,
        label_keys=("first", "second"),
    )
    encoded = {
        "train": {"labels": [[1.0, 10.0], [3.0, 14.0]]},
        "val": {"labels": [[5.0, 18.0]]},
        "test": {"labels": [[0.0, 8.0]]},
    }

    normalizer, output_dim = protocol._prepare_labels(encoded, spec, None)

    assert output_dim == 2
    assert normalizer == {
        "mean": [2.0, 12.0],
        "std": [1.0, 2.0],
        "target_properties": ["first", "second"],
    }
    assert encoded["train"]["labels"] == [[-1.0, -1.0], [1.0, 1.0]]
    assert encoded["val"]["labels"] == [[3.0, 3.0]]
    assert encoded["test"]["labels"] == [[-2.0, -2.0]]


def test_qm9_denormalization_restores_each_target_on_tensor_device():
    protocol = _load_paper_protocol_module()
    normalized = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    normalizer = {
        "mean": [2.0, 12.0],
        "std": [1.0, 2.0],
        "target_properties": ["first", "second"],
    }

    restored = protocol._denormalize_labels(torch, normalized, normalizer)

    assert restored.device == normalized.device
    assert restored.dtype == normalized.dtype
    assert torch.equal(restored, torch.tensor([[1.0, 10.0], [3.0, 14.0]]))


def test_qm9_prepare_labels_rejects_incorrect_target_width():
    protocol = _load_paper_protocol_module()
    spec = SimpleNamespace(
        canonical_name="qm9",
        output_dim=2,
        label_keys=("first", "second"),
    )
    encoded = {
        "train": {"labels": [[1.0, 10.0], [3.0, 14.0]]},
        "val": {"labels": [[5.0]]},
        "test": {"labels": [[0.0, 8.0]]},
    }

    with pytest.raises(ValueError, match="exactly 2 targets"):
        protocol._prepare_labels(encoded, spec, None)


def test_qm9_prepare_labels_rejects_nonfinite_targets():
    protocol = _load_paper_protocol_module()
    spec = SimpleNamespace(
        canonical_name="qm9",
        output_dim=2,
        label_keys=("first", "second"),
    )
    encoded = {
        "train": {"labels": [[1.0, 10.0], [3.0, float("nan")]]},
        "val": {"labels": [[5.0, 18.0]]},
        "test": {"labels": [[0.0, 8.0]]},
    }

    with pytest.raises(ValueError, match="non-finite target"):
        protocol._prepare_labels(encoded, spec, None)


def test_paper_protocol_uses_five_seeds_from_42_when_runs_is_unspecified():
    protocol = _load_paper_protocol_module()
    args = SimpleNamespace(dataset="qm9", target_property=None, runs=None, seed=None)

    protocol.validate_paper_args(args, QM9_SPEC)

    assert protocol.paper_run_seeds(args) == [42, 43, 44, 45, 46]


def test_paper_training_options_are_read_from_argparse_namespace():
    protocol = _load_paper_protocol_module()
    args = SimpleNamespace(
        model="gte",
        serialization="feuler",
        pretrain_epoch=200,
        n_epoch=200,
        pretrain_lr=5e-5,
        lr=1e-5,
        batch_size=32,
        weight_decay=0.1,
        pretrain_warmup_ratio=0.12,
        finetune_warmup_ratio=0.025,
        pretrain_max_grad_norm=2.0,
        finetune_max_grad_norm=0.5,
        mask_prob=0.09,
        patience=20,
        max_position_embeddings=8192,
        paper_amp="bf16",
        paper_tf32=True,
        paper_loss="huber",
        molhiv_pos_weight=True,
    )

    options = protocol.paper_training_options(args)

    assert options == {
        "encoder": "gte",
        "serialization": "feuler",
        "pretrain_epochs": 200,
        "finetune_epochs": 200,
        "pretrain_lr": 5e-5,
        "finetune_lr": 1e-5,
        "batch_size": 32,
        "weight_decay": 0.1,
        "pretrain_warmup_ratio": 0.12,
        "finetune_warmup_ratio": 0.025,
        "pretrain_max_grad_norm": 2.0,
        "finetune_max_grad_norm": 0.5,
        "mask_prob": 0.09,
        "patience": 20,
        "max_position_embeddings": 8192,
        "amp_dtype": "bf16",
        "allow_tf32": True,
        "training_loss": "huber",
        "molhiv_pos_weight": True,
    }


def test_paper_protocol_test_metrics_are_emitted_only_after_best_checkpoint():
    protocol = _load_paper_protocol_module()
    history = [
        {"epoch": 1, "val_metric": 0.4},
        {"epoch": 2, "val_metric": 0.2},
        {"epoch": 3, "val_metric": 0.3},
    ]

    assert protocol.select_best_epoch(history, higher_is_better=False) == 2
    assert "test" not in history[0]
    assert "test" not in history[1]
    assert "test" not in history[2]


def test_paper_regression_loss_backpropagates_from_half_precision_logits():
    protocol = _load_paper_protocol_module()
    logits = torch.tensor([[1.0]], dtype=torch.float16, requires_grad=True)
    labels = torch.tensor([[0.0]], dtype=torch.float32)

    loss = protocol._loss(torch, logits, labels, QM9_SPEC)
    loss.backward()

    assert logits.grad is not None


def test_optional_regression_losses_leave_paper_default_unchanged():
    protocol = _load_paper_protocol_module()
    spec = SimpleNamespace(canonical_name="peptides-struct")
    logits = torch.tensor([[3.0, 2.0]])
    labels = torch.tensor([[0.0, float("nan")]])

    default = protocol._loss(torch, logits, labels, spec)
    l1 = protocol._loss(torch, logits, labels, spec, loss_name="l1")
    huber = protocol._loss(torch, logits, labels, spec, loss_name="huber")

    assert default.item() == 9.0
    assert l1.item() == 3.0
    assert huber.item() == 2.5


def test_optional_molhiv_positive_weight_changes_positive_example_loss():
    protocol = _load_paper_protocol_module()
    spec = SimpleNamespace(canonical_name="molhiv")
    logits = torch.tensor([[0.0]])
    labels = torch.tensor([[1.0]])

    unweighted = protocol._loss(torch, logits, labels, spec)
    weighted = protocol._loss(
        torch, logits, labels, spec, pos_weight=torch.tensor(3.0))

    assert weighted.item() == pytest.approx(unweighted.item() * 3.0)


def test_precision_options_reject_cuda_amp_on_cpu():
    protocol = _load_paper_protocol_module()

    assert protocol._resolve_precision_options(
        torch, torch.device("cpu"), {"amp_dtype": "off"}) == {
            "amp_dtype": "off",
            "torch_dtype": None,
            "grad_scaler": None,
            "allow_tf32": False,
        }
    with pytest.raises(ValueError, match="CUDA device"):
        protocol._resolve_precision_options(
            torch, torch.device("cpu"), {"amp_dtype": "fp16"})


def test_epoch_runtime_metrics_report_cpu_throughput():
    protocol = _load_paper_protocol_module()

    metrics = protocol._epoch_runtime_metrics(
        torch, torch.device("cpu"), num_examples=10, seconds=2.0)

    assert metrics == {
        "seconds": 2.0,
        "examples_per_second": 5.0,
    }


def test_paper_metric_details_report_each_qm9_target_mae():
    protocol = _load_paper_protocol_module()
    spec = SimpleNamespace(
        canonical_name="qm9",
        label_keys=("first", "second"),
    )

    details = protocol.compute_paper_metric_details(
        spec,
        [[1.0, 4.0], [3.0, 8.0]],
        [[2.0, 2.0], [1.0, 9.0]],
    )

    assert details == {
        "metric": 1.5,
        "per_target_mae": {"first": 1.5, "second": 1.5},
    }


def test_paper_mlm_mask_forces_one_token_when_random_selection_is_empty(monkeypatch):
    protocol = _load_paper_protocol_module()
    tokenizer = SimpleNamespace(special_tokens=SimpleNamespace(
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        component_sep_token_id=3,
        mask_token_id=4,
    ))
    input_ids = torch.tensor([[1, 5, 6, 3, 7, 2, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0]])
    monkeypatch.setattr(
        torch,
        "rand_like",
        lambda tensor, dtype: torch.ones_like(tensor, dtype=dtype),
    )

    masked_ids, labels = protocol._mask_for_mlm(
        torch, input_ids, attention_mask, tokenizer, mask_prob=0.09)

    selected = labels.ne(-100)
    maskable = torch.tensor([[False, True, True, False, True, False, False]])
    assert selected.sum().item() == 1
    assert torch.all(selected <= maskable)
    assert torch.equal(labels[selected], input_ids[selected])
    assert torch.all(masked_ids[selected].eq(tokenizer.special_tokens.mask_token_id))
    assert torch.equal(masked_ids[~selected], input_ids[~selected])


def test_mlm_training_switches_tensorlayerx_model_to_train_mode():
    protocol = _load_paper_protocol_module()

    class TLXModeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.is_train = False

        def set_train(self):
            self.is_train = True

        def forward(self, input_ids, attention_mask, task):
            return self.weight.expand(len(input_ids), input_ids.shape[1], 8)

    tokenizer = SimpleNamespace(special_tokens=SimpleNamespace(
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        component_sep_token_id=3,
        mask_token_id=4,
    ))
    model = TLXModeModel()
    loader = [(
        torch.tensor([[1, 5, 2]]),
        torch.tensor([[1, 1, 1]]),
        torch.tensor([[0.0]]),
    )]
    protocol._train_mlm(
        torch,
        model,
        loader,
        torch.optim.SGD(model.parameters(), lr=0.0),
        SimpleNamespace(step=lambda: None),
        tokenizer,
        torch.device("cpu"),
        max_grad_norm=10.0,
        mask_prob=0.09,
    )

    assert model.is_train is True


def test_paper_evaluation_rejects_nonfinite_predictions():
    protocol = _load_paper_protocol_module()

    class NonFiniteModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask, task):
            return torch.full((len(input_ids), 1), float("nan"))

    loader = [(torch.tensor([[1]]), torch.tensor([[1]]), torch.tensor([[0.0]]))]

    with pytest.raises(FloatingPointError, match="supervised logits"):
        protocol._evaluate(torch, NonFiniteModel(), loader, QM9_SPEC, "cpu", None)


def test_paper_evaluation_switches_tensorlayerx_model_to_eval_mode():
    """Evaluation must disable TLX Dropout, which reads ``is_train``."""
    protocol = _load_paper_protocol_module()

    class TLXModeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.is_train = True

        def set_eval(self):
            self.is_train = False

        def forward(self, input_ids, attention_mask, task):
            return torch.zeros((len(input_ids), 1), dtype=torch.float32)

    model = TLXModeModel()
    loader = [(torch.tensor([[1]]), torch.tensor([[1]]), torch.tensor([[0.0]]))]

    protocol._evaluate(
        torch,
        model,
        loader,
        SimpleNamespace(canonical_name="qm9", label_keys=("target",)),
        torch.device("cpu"),
        normalizer=None,
    )

    assert model.is_train is False


def test_amp_evaluation_casts_predictions_to_fp32_before_denormalizing(
        monkeypatch):
    protocol = _load_paper_protocol_module()

    class HalfModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask, task):
            return torch.full(
                (len(input_ids), 1), 0.3333, dtype=torch.float16)

    loader = [(
        torch.tensor([[1]]),
        torch.tensor([[1]]),
        torch.tensor([[0.0]], dtype=torch.float32),
    )]
    dtypes = []

    def record_dtype(_torch, values, _normalizer):
        dtypes.append(values.dtype)
        return values

    monkeypatch.setattr(protocol, "_denormalize_labels", record_dtype)
    protocol._evaluate(
        torch,
        HalfModel(),
        loader,
        SimpleNamespace(canonical_name="qm9", label_keys=("target",)),
        torch.device("cpu"),
        normalizer={"mean": [0.0], "std": [1.0]},
    )

    assert dtypes == [torch.float32, torch.float32]


def test_qm9_evaluation_uses_standardized_mae_and_keeps_raw_diagnostics():
    protocol = _load_paper_protocol_module()

    class ZeroModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask, task):
            return torch.zeros((len(input_ids), 2), dtype=torch.float32)

    result = protocol._evaluate(
        torch,
        ZeroModel(),
        [(torch.tensor([[1]]), torch.tensor([[1]]),
          torch.tensor([[1.0, 1.0]]))],
        SimpleNamespace(
            canonical_name="qm9", label_keys=("first", "second")),
        torch.device("cpu"),
        normalizer={"mean": [100.0, 1000.0], "std": [10.0, 100.0]},
    )

    assert result["metric"] == 55.0
    assert result["metric_space"] == "raw"
    assert result["per_target_mae"] == {"first": 1.0, "second": 1.0}
    assert result["per_target_mae_raw"] == {"first": 10.0, "second": 100.0}


def test_paper_encoding_keeps_ragged_sequences_until_batch_collation():
    protocol = _load_paper_protocol_module()

    class Tokenizer:
        special_tokens = SimpleNamespace(pad_token_id=0)

        @staticmethod
        def encode_graph(graph):
            return SimpleNamespace(input_ids=list(graph.tokens))

    def graph(tokens, label):
        return SimpleNamespace(tokens=tokens, y=[label])

    splits = {
        "train": [graph([3, 8, 4], 1.0), graph([3, 9, 10, 11, 4], 2.0)],
        "val": [graph([3, 4], 3.0)],
        "test": [graph([3, 12, 4], 4.0)],
    }

    encoded, maximum = protocol._encode_splits(
        Tokenizer(), splits, max_position_embeddings=16)

    assert maximum == 5
    assert encoded["train"]["input_ids"] == [
        [3, 8, 4],
        [3, 9, 10, 11, 4],
    ]
    assert "attention_mask" not in encoded["train"]


def test_paper_encoded_splits_reuse_local_cache_without_reserializing(tmp_path):
    protocol = _load_paper_protocol_module()

    class CountingTokenizer:
        calls = 0

        @classmethod
        def encode_graph(cls, graph):
            cls.calls += 1
            return SimpleNamespace(input_ids=list(graph.tokens))

    def graph(tokens, label):
        return SimpleNamespace(tokens=tokens, y=[label])

    splits = {
        "train": [graph([3, 8, 4], 1.0)],
        "val": [graph([3, 9, 4], 2.0)],
        "test": [graph([3, 10, 4], 3.0)],
    }
    cache_path = tmp_path / "encoded.pkl"

    first, first_max = protocol._load_or_encode_splits(
        CountingTokenizer(), splits, 16, cache_path, cache_key="fixture-v1")
    second, second_max = protocol._load_or_encode_splits(
        CountingTokenizer(), splits, 16, cache_path, cache_key="fixture-v1")

    assert CountingTokenizer.calls == 3
    assert second == first
    assert second_max == first_max == 3


def test_encoded_cache_key_changes_when_validation_content_changes():
    protocol = _load_paper_protocol_module()
    tokenizer = SimpleNamespace(
        _cache_key="train-v1",
        serializer=SimpleNamespace(name="feuler", frequency_map={}),
        bpe=SimpleNamespace(codebook=SimpleNamespace(
            merge_rules=[], vocab_size=8)),
    )

    def graph(label):
        return SimpleNamespace(
            edge_index=[[0], [1]],
            x=[13, 17],
            edge_attr=[2],
            num_nodes=2,
            y=[label],
        )

    first_splits = {
        "train": [graph(0.0)],
        "val": [graph(1.0)],
        "test": [graph(2.0)],
    }
    second_splits = {
        **first_splits,
        "val": [graph(99.0)],
    }

    first = protocol._encoded_splits_cache_key(tokenizer, first_splits, 16)
    second = protocol._encoded_splits_cache_key(tokenizer, second_splits, 16)

    assert first != second


def test_paper_encoding_chunks_native_batch_calls():
    protocol = _load_paper_protocol_module()

    class Tokenizer:
        calls = []

        @classmethod
        def batch_encode_graphs(cls, graphs):
            cls.calls.append(len(graphs))
            return [SimpleNamespace(input_ids=list(graph.tokens)) for graph in graphs]

    graphs = [
        SimpleNamespace(tokens=[3, index, 4], y=[float(index)])
        for index in range(5)
    ]
    splits = {"train": graphs, "val": graphs[:1], "test": graphs[:1]}

    encoded, maximum = protocol._encode_splits(
        Tokenizer(), splits, 16, encoding_batch_size=2)

    assert Tokenizer.calls == [2, 2, 1, 1, 1]
    assert len(encoded["train"]["input_ids"]) == 5
    assert maximum == 3


def test_paper_loader_pads_only_to_each_batch_maximum():
    protocol = _load_paper_protocol_module()
    split = {
        "input_ids": [[3, 8, 4], [3, 9, 10, 11, 4], [3, 4]],
        "labels": [[1.0], [2.0], [3.0]],
    }

    loader = protocol._make_loader(
        torch,
        split,
        batch_size=2,
        shuffle=False,
        pad_token_id=0,
        num_workers=0,
        pin_memory=False,
    )
    first_ids, first_mask, first_labels = next(iter(loader))
    batches = list(loader)
    second_ids, second_mask, second_labels = batches[1]

    assert first_ids.tolist() == [[3, 8, 4, 0, 0], [3, 9, 10, 11, 4]]
    assert first_mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert first_labels.tolist() == [[1.0], [2.0]]
    assert second_ids.tolist() == [[3, 4]]
    assert second_mask.tolist() == [[1, 1]]
    assert second_labels.tolist() == [[3.0]]


def test_dataloader_worker_seed_does_not_consume_model_rng():
    protocol = _load_paper_protocol_module()
    split = {
        "input_ids": [[3, 4]],
        "labels": [[1.0]],
    }
    torch.manual_seed(123)
    expected = torch.rand(1)
    torch.manual_seed(123)
    loader = protocol._make_loader(
        torch,
        split,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        bucket_by_length=False,
    )

    next(iter(loader))
    actual = torch.rand(1)

    assert torch.equal(actual, expected)


def test_paper_memory_plan_keeps_effective_batch_for_short_sequences():
    protocol = _load_paper_protocol_module()

    plan = protocol._resolve_memory_plan(
        encoder_type="gte",
        effective_batch_size=16,
        sequence_length=165,
        available_bytes=24 * 1024 ** 3,
    )

    assert plan["micro_batch_size"] == 16
    assert plan["gradient_accumulation_steps"] == 1
    assert plan["effective_batch_size"] == 16
    assert plan["estimated_peak_bytes"] < plan["budget_bytes"]


def test_cuda_memory_query_uses_current_index_for_bare_cuda_device(monkeypatch):
    protocol = _load_paper_protocol_module()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    assert protocol._cuda_memory_query_device(torch, torch.device("cuda")) == 3
    assert (
        protocol._cuda_memory_query_device(torch, torch.device("cuda:2"))
        == torch.device("cuda:2")
    )


def test_paper_memory_plan_uses_gradient_accumulation_when_needed():
    protocol = _load_paper_protocol_module()

    plan = protocol._resolve_memory_plan(
        encoder_type="bert",
        effective_batch_size=32,
        sequence_length=768,
        available_bytes=6 * 1024 ** 3,
    )

    assert 1 <= plan["micro_batch_size"] < 32
    assert 32 % plan["micro_batch_size"] == 0
    assert (
        plan["micro_batch_size"]
        * plan["gradient_accumulation_steps"]
        == 32
    )


def test_paper_memory_plan_accounts_for_mixed_precision_activations():
    protocol = _load_paper_protocol_module()

    fp32 = protocol._estimate_training_peak_bytes(
        "bert", micro_batch_size=8, sequence_length=256,
        activation_bytes_per_element=4)
    mixed = protocol._estimate_training_peak_bytes(
        "bert", micro_batch_size=8, sequence_length=256,
        activation_bytes_per_element=2)

    assert mixed < fp32


def test_resume_skips_finetuning_when_saved_state_already_hit_patience():
    protocol = _load_paper_protocol_module()

    assert protocol._should_skip_finetuning(
        resume_phase="finetune", stale_epochs=20, patience=20)
    assert protocol._should_skip_finetuning(
        resume_phase="finetune_complete", stale_epochs=0, patience=20)
    assert not protocol._should_skip_finetuning(
        resume_phase="finetune", stale_epochs=19, patience=20)


def test_paper_memory_plan_rejects_impossible_dense_attention():
    protocol = _load_paper_protocol_module()

    with pytest.raises(RuntimeError, match="dense attention"):
        protocol._resolve_memory_plan(
            encoder_type="gte",
            effective_batch_size=16,
            sequence_length=8192,
            available_bytes=24 * 1024 ** 3,
        )


def test_supervised_training_accumulates_to_the_effective_batch():
    protocol = _load_paper_protocol_module()

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[0.5]]))

        def forward(self, input_ids, attention_mask, task):
            assert task == "supervised"
            return input_ids.float()[:, :1] @ self.weight

    model = TinyModel()
    loader = [
        (torch.tensor([[value]]), torch.ones((1, 1), dtype=torch.long),
         torch.tensor([[0.0]]))
        for value in (1, 2, 3, 4)
    ]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    step_calls = []
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        step_calls.append(True)
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    scheduler = SimpleNamespace(step=lambda: None)

    protocol._train_supervised(
        torch,
        model,
        loader,
        optimizer,
        scheduler,
        QM9_SPEC,
        "cpu",
        max_grad_norm=10.0,
        gradient_accumulation_steps=2,
    )

    assert len(step_calls) == 2


def test_supervised_training_uses_explicit_regression_loss():
    protocol = _load_paper_protocol_module()

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[3.0]]))

        def forward(self, input_ids, attention_mask, task):
            return input_ids.float()[:, :1] @ self.weight

    model = TinyModel()
    loader = [(
        torch.tensor([[1]]),
        torch.ones((1, 1), dtype=torch.long),
        torch.tensor([[0.0]]),
    )]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    loss = protocol._train_supervised(
        torch,
        model,
        loader,
        optimizer,
        SimpleNamespace(step=lambda: None),
        SimpleNamespace(canonical_name="qm9"),
        torch.device("cpu"),
        max_grad_norm=10.0,
        loss_name="l1",
    )

    assert loss == 3.0


def test_supervised_training_switches_tensorlayerx_model_to_train_mode():
    protocol = _load_paper_protocol_module()

    class TLXModeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[1.0]]))
            self.is_train = False

        def set_train(self):
            self.is_train = True

        def forward(self, input_ids, attention_mask, task):
            return input_ids.float()[:, :1] @ self.weight

    model = TLXModeModel()
    loader = [(
        torch.tensor([[1]]), torch.ones((1, 1), dtype=torch.long),
        torch.tensor([[0.0]]),
    )]
    protocol._train_supervised(
        torch,
        model,
        loader,
        torch.optim.SGD(model.parameters(), lr=0.0),
        SimpleNamespace(step=lambda: None),
        SimpleNamespace(canonical_name="qm9"),
        torch.device("cpu"),
        max_grad_norm=10.0,
    )

    assert model.is_train is True


def test_gradient_accumulation_weights_uneven_microbatches_by_examples():
    protocol = _load_paper_protocol_module()

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[1.0]]))

        def forward(self, input_ids, attention_mask, task):
            return input_ids.float()[:, :1] @ self.weight

    model = TinyModel()
    loader = [
        (
            torch.tensor([[1], [1]]),
            torch.ones((2, 1), dtype=torch.long),
            torch.zeros((2, 1)),
        ),
        (
            torch.tensor([[3]]),
            torch.ones((1, 1), dtype=torch.long),
            torch.zeros((1, 1)),
        ),
    ]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    gradients = []
    original_step = optimizer.step

    def record_gradient(*args, **kwargs):
        gradients.append(float(model.weight.grad.item()))
        return original_step(*args, **kwargs)

    optimizer.step = record_gradient

    protocol._train_supervised(
        torch,
        model,
        loader,
        optimizer,
        SimpleNamespace(step=lambda: None),
        SimpleNamespace(canonical_name="qm9"),
        torch.device("cpu"),
        max_grad_norm=100.0,
        gradient_accumulation_steps=2,
    )

    assert gradients == pytest.approx([22.0 / 3.0])


def test_fp16_grad_scaler_is_new_for_each_independent_run():
    protocol = _load_paper_protocol_module()

    class FakeGradScaler:
        pass

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            amp=SimpleNamespace(GradScaler=lambda enabled: FakeGradScaler())))

    first = protocol._new_grad_scaler(fake_torch, "fp16")
    second = protocol._new_grad_scaler(fake_torch, "fp16")

    assert isinstance(first, FakeGradScaler)
    assert isinstance(second, FakeGradScaler)
    assert first is not second
    assert protocol._new_grad_scaler(fake_torch, "bf16") is None


def test_fp16_overflow_skips_optimizer_and_scheduler_step():
    protocol = _load_paper_protocol_module()

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[1.0]]))

        def forward(self, input_ids, attention_mask, task):
            return input_ids.float()[:, :1] @ self.weight

    model = TinyModel()
    loader = [(
        torch.tensor([[1]]),
        torch.ones((1, 1), dtype=torch.long),
        torch.zeros((1, 1)),
    )]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer_steps = []
    scheduler_steps = []
    optimizer.step = lambda *_args, **_kwargs: optimizer_steps.append(True)

    class OverflowScaler:
        scale_value = 8.0

        @staticmethod
        def scale(loss):
            return loss

        @staticmethod
        def unscale_(_optimizer):
            model.weight.grad.fill_(float("inf"))

        @staticmethod
        def step(_optimizer):
            return None

        def update(self):
            self.scale_value /= 2

        def get_scale(self):
            return self.scale_value

    protocol._train_supervised(
        torch,
        model,
        loader,
        optimizer,
        SimpleNamespace(step=lambda: scheduler_steps.append(True)),
        SimpleNamespace(canonical_name="qm9"),
        torch.device("cpu"),
        max_grad_norm=1.0,
        grad_scaler=OverflowScaler(),
    )

    assert optimizer_steps == []
    assert scheduler_steps == []


def test_molhiv_positive_weight_uses_training_labels_only():
    protocol = _load_paper_protocol_module()
    encoded_train = {
        "labels": [[0.0], [0.0], [0.0], [1.0]],
    }

    weight = protocol._molhiv_positive_weight(torch, encoded_train, "cpu")

    assert weight.item() == 3.0


def test_best_checkpoint_is_lightweight_and_restores_model(tmp_path):
    protocol = _load_paper_protocol_module()
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "best.pt"

    protocol.save_paper_checkpoint(
        path,
        torch,
        model,
        epoch=7,
        best_metric=0.25,
        normalizer={"mean": [1.0], "std": [2.0]},
    )
    state = protocol._torch_load(torch, path, device="cpu")

    assert set(state) == {
        "checkpoint_kind",
        "model",
        "epoch",
        "best_metric",
        "normalizer",
    }
    assert state["checkpoint_kind"] == "best_model"
    assert "optimizer" not in state
    assert "scheduler" not in state
    with torch.no_grad():
        model.weight.zero_()
    restored = protocol.restore_paper_checkpoint(
        path, torch, model, device="cpu")
    assert restored["epoch"] == 7
    assert torch.count_nonzero(model.weight).item() > 0


def test_checkpoint_rejects_a_different_experiment_fingerprint(tmp_path):
    protocol = _load_paper_protocol_module()
    model = torch.nn.Linear(1, 1)
    path = tmp_path / "best.pt"
    protocol.save_paper_checkpoint(
        path,
        torch,
        model,
        epoch=1,
        best_metric=0.5,
        normalizer=None,
        experiment_fingerprint="experiment-a",
    )

    with pytest.raises(ValueError, match="fingerprint"):
        protocol.restore_paper_checkpoint(
            path,
            torch,
            model,
            device="cpu",
            expected_fingerprint="experiment-b",
        )


def test_resume_checkpoint_roundtrips_training_and_rng_state(tmp_path):
    protocol = _load_paper_protocol_module()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda _: 1.0)
    loss = model(torch.ones((1, 2))).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    path = tmp_path / "last_state.pt"

    protocol.save_paper_resume_state(
        path,
        torch,
        model,
        optimizer,
        scheduler,
        phase="pretrain",
        epoch=3,
        extra={"history": [{"epoch": 1}]},
    )
    state = protocol._torch_load(torch, path, device="cpu")

    assert state["checkpoint_kind"] == "resume_state"
    assert state["phase"] == "pretrain"
    assert state["epoch"] == 3
    assert "optimizer" in state
    assert "scheduler" in state
    assert "rng_state" in state
    restored = protocol.restore_paper_resume_state(
        path, torch, model, optimizer, scheduler, device="cpu")
    assert restored["extra"]["history"] == [{"epoch": 1}]


def test_resume_checkpoint_roundtrips_amp_scaler_state(tmp_path):
    protocol = _load_paper_protocol_module()
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda _: 1.0)

    class FakeScaler:
        def __init__(self, scale):
            self.scale = scale

        def state_dict(self):
            return {"scale": self.scale}

        def load_state_dict(self, state):
            self.scale = state["scale"]

    path = tmp_path / "amp_state.pt"
    protocol.save_paper_resume_state(
        path,
        torch,
        model,
        optimizer,
        scheduler,
        phase="pretrain",
        epoch=2,
        grad_scaler=FakeScaler(1024.0),
    )
    restored_scaler = FakeScaler(1.0)

    protocol.restore_paper_resume_state(
        path,
        torch,
        model,
        optimizer,
        scheduler,
        device="cpu",
        grad_scaler=restored_scaler,
    )

    assert restored_scaler.scale == 1024.0


def test_pretrain_complete_state_restores_amp_scaler_before_finetuning():
    protocol = _load_paper_protocol_module()

    class FakeScaler:
        scale = 1.0

        def load_state_dict(self, state):
            self.scale = state["scale"]

    scaler = FakeScaler()
    protocol._restore_grad_scaler_state(
        {"grad_scaler": {"scale": 512.0}}, scaler)

    assert scaler.scale == 512.0


def test_run_fingerprint_changes_with_seed_and_run_index():
    protocol = _load_paper_protocol_module()
    training_options = {"encoder": "gte", "batch_size": 32}
    spec = SimpleNamespace(
        canonical_name="qm9",
        task_type="regression",
        output_dim=16,
    )

    first = protocol._experiment_fingerprint(
        training_options, spec, "encoded", 100, "mean", seed=42, run_index=0)
    different_seed = protocol._experiment_fingerprint(
        training_options, spec, "encoded", 100, "mean", seed=43, run_index=0)
    different_run = protocol._experiment_fingerprint(
        training_options, spec, "encoded", 100, "mean", seed=42, run_index=1)

    assert len({first, different_seed, different_run}) == 3


def test_tiny_paper_protocol_runs_pretrain_finetune_and_five_runs(
        tmp_path, monkeypatch):
    protocol = _load_paper_protocol_module()

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 4)
            self.task_head = torch.nn.Linear(4, 16)
            self.mlm_head = torch.nn.Linear(4, 32)

        def forward(self, input_ids, attention_mask, task):
            hidden = self.embedding(input_ids)
            if task == "mlm":
                return self.mlm_head(hidden)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            return self.task_head(pooled)

        @staticmethod
        def manifest():
            return {"encoder_type": "bert", "tiny": True}

    class Tokenizer:
        special_tokens = SimpleNamespace(
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
            component_sep_token_id=3,
            mask_token_id=4,
        )
        serializer = SimpleNamespace(name="feuler", frequency_map={})
        bpe = SimpleNamespace(codebook=SimpleNamespace(
            merge_rules=[], vocab_size=32))
        max_token_id = 31
        _cache_key = "tiny-tokenizer"
        _cache_status = "miss"
        _cache_path = str(tmp_path / "tokenizer.pkl")

        @staticmethod
        def validate_model_vocab(vocab_size):
            assert vocab_size == 32

        @staticmethod
        def encode_graph(graph):
            return SimpleNamespace(input_ids=list(graph.tokens))

    labels = [float(index) for index in range(16)]

    def graph(token, offset):
        return SimpleNamespace(
            tokens=[1, token, 2],
            y=[value + offset for value in labels],
        )

    splits = {
        "train": [graph(5, 0.0), graph(6, 1.0)],
        "val": [graph(7, 0.5)],
        "test": [graph(8, 0.25)],
    }
    args = SimpleNamespace(
        dataset="qm9",
        target_property=None,
        runs=5,
        seed=42,
        model="bert",
        pooling="mean",
        device="cpu",
        num_workers=0,
        output_dir=str(tmp_path / "output"),
        data_root=str(tmp_path / "data"),
        paper_cache_root=str(tmp_path / "cache"),
        resume=False,
    )
    training_options = {
        "encoder": "bert",
        "serialization": "feuler",
        "pretrain_epochs": 1,
        "finetune_epochs": 3,
        "pretrain_lr": 1e-4,
        "finetune_lr": 1e-5,
        "patience": 0,
        "batch_size": 2,
        "weight_decay": 0.1,
        "pretrain_warmup_ratio": 0.12,
        "finetune_warmup_ratio": 0.025,
        "pretrain_max_grad_norm": 2.0,
        "finetune_max_grad_norm": 0.5,
        "mask_prob": 0.09,
        "max_position_embeddings": 16,
    }
    monkeypatch.setattr(protocol, "_torch", lambda: torch)
    monkeypatch.setattr(
        protocol, "paper_training_options", lambda _args: training_options)
    monkeypatch.setattr(protocol, "_create_paper_model", lambda **_kwargs: TinyModel())
    scaler_runs = []
    monkeypatch.setattr(
        protocol,
        "_new_grad_scaler",
        lambda _torch, amp_dtype: scaler_runs.append(amp_dtype) or None,
    )

    summary = protocol.run_paper_experiment(
        args, QM9_SPEC, splits, lambda _args, _graphs: Tokenizer())

    assert len(summary["runs"]) == 5
    assert scaler_runs == ["off"] * 5
    assert all(run["test_evaluations"] == 1 for run in summary["runs"])
    assert all(Path(run["checkpoint_path"]).is_file() for run in summary["runs"])
    assert summary["cache"]["encoded_status"] == "miss"
    assert summary["precision"] == {
        "amp_dtype": "off",
        "allow_tf32": False,
    }

    args.resume = True

    def fail_if_completed_run_retrains(*_args, **_kwargs):
        raise AssertionError("a finetune_complete run must not train again")

    monkeypatch.setattr(
        protocol, "_train_supervised", fail_if_completed_run_retrains)
    resumed = protocol.run_paper_experiment(
        args, QM9_SPEC, splits, lambda _args, _graphs: Tokenizer())

    assert all(run["resumed"] for run in resumed["runs"])
