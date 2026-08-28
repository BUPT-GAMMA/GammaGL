import importlib.util
import json
import pickle
import gzip
import sys
import tempfile
import types
from types import SimpleNamespace
from pathlib import Path

import pytest
import tensorlayerx as tlx


torch = pytest.importorskip("torch")


def _patch_tlx_torch_named_members():
    import tensorlayerx.nn.core.core_torch as core_torch

    if getattr(core_torch.Module._named_members, "_graph_tokenizer_patch", False):
        return

    original = core_torch.Module._named_members

    def patched(self, get_members_fn, prefix="", recurse=True, remove_duplicate=True):
        return original(self, get_members_fn, prefix=prefix, recurse=recurse)

    patched._graph_tokenizer_patch = True
    core_torch.Module._named_members = patched


def _load_model_module(filename, module_name):
    module_path = Path(__file__).resolve().parents[2] / "gammagl" / "models" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_trainer_module():
    module_path = Path(__file__).resolve().parents[2] / "examples" / "graph_tokenizer" / "graph_tokenizer_trainer.py"
    spec = importlib.util.spec_from_file_location("graph_tokenizer_trainer_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _qm9_properties(offset=0.0):
    names = (
        "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0",
        "u298", "h298", "g298", "cv", "u0_atom", "u298_atom",
        "h298_atom", "g298_atom",
    )
    return {name: offset + (index + 1) / 10.0 for index, name in enumerate(names)}


def test_graph_transformer_public_interfaces_are_importable():
    GraphBERT = _load_model_module("graph_bert.py", "graph_bert_public_import_test").GraphBERT
    _load_model_module("graph_bert.py", "graph_bert")
    GraphGTE = _load_model_module("graph_gte.py", "graph_gte_public_import_test").GraphGTE

    bert = GraphBERT(vocab_size=32, output_dim=2)
    gte = GraphGTE(vocab_size=32, output_dim=2)

    assert bert.vocab_size == 32
    assert bert.output_dim == 2
    assert gte.vocab_size == 32
    assert gte.output_dim == 2


def test_graph_bert_forward_returns_task_and_mlm_logits():
    _patch_tlx_torch_named_members()
    GraphBERT = _load_model_module("graph_bert.py", "graph_bert_under_test").GraphBERT

    model = GraphBERT(
        vocab_size=32,
        output_dim=3,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    input_ids = tlx.convert_to_tensor([[3, 5, 6, 4], [3, 7, 0, 0]], dtype=tlx.int64)
    attention_mask = tlx.convert_to_tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=tlx.int64)

    outputs = model(input_ids, attention_mask=attention_mask)

    assert tlx.get_tensor_shape(outputs["logits"]) == [2, 3]
    assert tlx.get_tensor_shape(outputs["mlm_logits"]) == [2, 4, 32]
    assert tlx.get_tensor_shape(outputs["last_hidden_state"]) == [2, 4, 16]
    assert tlx.get_tensor_shape(outputs["pooled_output"]) == [2, 16]


def test_graph_bert_supports_paper_task_outputs():
    _patch_tlx_torch_named_members()
    GraphBERT = _load_model_module(
        "graph_bert.py", "graph_bert_paper_outputs_test").GraphBERT
    model = GraphBERT(
        vocab_size=32,
        output_dim=3,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    input_ids = tlx.convert_to_tensor([[3, 5, 6, 4]], dtype=tlx.int64)
    attention_mask = tlx.convert_to_tensor([[1, 1, 1, 1]], dtype=tlx.int64)

    mlm_logits = model(input_ids, attention_mask=attention_mask, task="mlm")
    task_logits = model(input_ids, attention_mask=attention_mask, task="supervised")

    assert tlx.get_tensor_shape(mlm_logits) == [1, 4, 32]
    assert tlx.get_tensor_shape(task_logits) == [1, 3]


@pytest.mark.parametrize(
    ("filename", "module_name", "class_name"),
    [
        ("graph_bert.py", "graph_bert_task_routing_test", "GraphBERT"),
        ("graph_gte.py", "graph_gte_task_routing_test", "GraphGTE"),
    ],
)
def test_graph_transformers_only_compute_the_requested_output_head(
        filename, module_name, class_name, monkeypatch):
    _patch_tlx_torch_named_members()
    _load_model_module("graph_bert.py", "graph_bert")
    model_class = getattr(_load_model_module(filename, module_name), class_name)
    model = model_class(
        vocab_size=32,
        output_dim=2,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
        dropout_rate=0.0,
    )
    input_ids = tlx.convert_to_tensor([[3, 5, 4]], dtype=tlx.int64)
    attention_mask = tlx.convert_to_tensor([[1, 1, 1]], dtype=tlx.int64)

    def fail_mlm(*args, **kwargs):
        raise AssertionError("MLM head must not run for supervised inference")

    monkeypatch.setattr(model.mlm_head, "forward", fail_mlm)
    supervised = model(
        input_ids, attention_mask=attention_mask, task="supervised")
    assert tlx.get_tensor_shape(supervised) == [1, 2]

    monkeypatch.undo()

    def fail_task(*args, **kwargs):
        raise AssertionError("task head must not run for MLM inference")

    monkeypatch.setattr(model, "_task_logits", fail_task)
    mlm = model(input_ids, attention_mask=attention_mask, task="mlm")
    assert tlx.get_tensor_shape(mlm) == [1, 3, 32]


def test_graph_bert_padding_tokens_do_not_change_supervised_output():
    _patch_tlx_torch_named_members()
    GraphBERT = _load_model_module(
        "graph_bert.py", "graph_bert_padding_test").GraphBERT
    model = GraphBERT(
        vocab_size=32,
        output_dim=2,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
        dropout_rate=0.0,
    )
    model.set_eval()
    attention_mask = tlx.convert_to_tensor([[1, 1, 1, 0]], dtype=tlx.int64)
    first = tlx.convert_to_tensor([[3, 5, 4, 0]], dtype=tlx.int64)
    second = tlx.convert_to_tensor([[3, 5, 4, 17]], dtype=tlx.int64)

    first_logits = model(first, attention_mask=attention_mask, task="supervised")
    second_logits = model(second, attention_mask=attention_mask, task="supervised")

    assert torch.allclose(first_logits, second_logits, atol=1e-6, rtol=1e-6)


def test_graph_bert_manifest_reports_strict_paper_defaults():
    _patch_tlx_torch_named_members()
    GraphBERT = _load_model_module(
        "graph_bert.py", "graph_bert_manifest_test").GraphBERT
    model = GraphBERT(vocab_size=2200, output_dim=16)

    manifest = model.manifest()

    assert manifest["encoder_type"] == "bert"
    assert manifest["model_name"] == "bert-small"
    assert manifest["hidden_size"] == 512
    assert manifest["num_hidden_layers"] == 4
    assert manifest["num_attention_heads"] == 4
    assert manifest["intermediate_size"] == 2048
    assert manifest["hidden_act"] == "gelu"
    assert manifest["hidden_dropout_prob"] == 0.1
    assert manifest["attention_probs_dropout_prob"] == 0.1
    assert manifest["position_embedding_type"] == "absolute"
    assert manifest["max_position_embeddings"] == 768
    assert manifest["layer_norm_eps"] == 1e-12
    assert 14_000_000 < manifest["total_parameters"] < 20_000_000
    assert manifest["trainable_parameters"] == manifest["total_parameters"]


def test_graph_bert_backpropagates_through_attention_and_both_heads():
    _patch_tlx_torch_named_members()
    GraphBERT = _load_model_module(
        "graph_bert.py", "graph_bert_gradient_test").GraphBERT
    model = GraphBERT(
        vocab_size=24,
        output_dim=2,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=8,
        dropout_rate=0.0,
    )
    input_ids = tlx.convert_to_tensor([[3, 5, 6, 4]], dtype=tlx.int64)
    attention_mask = tlx.convert_to_tensor([[1, 1, 1, 1]], dtype=tlx.int64)

    loss = (
        model(input_ids, attention_mask=attention_mask, task="mlm").sum()
        + model(input_ids, attention_mask=attention_mask, task="supervised").sum()
    )
    loss.backward()

    checked = [
        model.encoder_layers[0].attention.q_proj.trainable_weights[0],
        model.mlm_head.trainable_weights[0],
        model.task_head_out.trainable_weights[0],
    ]
    for parameter in checked:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0


def test_graph_bert_attention_supports_nonsquare_sequence_and_head_shapes():
    _patch_tlx_torch_named_members()
    GraphBERT = _load_model_module(
        "graph_bert.py", "graph_bert_nonsquare_attention_test").GraphBERT
    model = GraphBERT(
        vocab_size=32,
        output_dim=1,
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=48,
        max_position_embeddings=16,
        dropout_rate=0.0,
    )
    input_ids = tlx.convert_to_tensor([[3, 5, 6, 7, 4]], dtype=tlx.int64)

    outputs = model(input_ids)

    assert tlx.get_tensor_shape(outputs["last_hidden_state"]) == [1, 5, 24]


def test_graph_gte_uses_long_position_window_and_forward_shape():
    _patch_tlx_torch_named_members()
    graph_bert = _load_model_module("graph_bert.py", "graph_bert")
    GraphGTE = _load_model_module("graph_gte.py", "graph_gte_under_test").GraphGTE

    model = GraphGTE(
        vocab_size=48,
        output_dim=1,
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=48,
        max_position_embeddings=64,
    )
    input_ids = tlx.convert_to_tensor([[3, 8, 9, 10, 4]], dtype=tlx.int64)

    outputs = model(input_ids)

    assert graph_bert.GraphBERT(vocab_size=48, output_dim=1).max_position_embeddings == 768
    assert model.max_position_embeddings == 64
    assert tlx.get_tensor_shape(outputs["logits"]) == [1, 1]
    assert tlx.get_tensor_shape(outputs["mlm_logits"]) == [1, 5, 48]


def test_graph_gte_applies_rotary_embedding_to_query_and_key():
    _load_model_module("graph_bert.py", "graph_bert")
    graph_gte = _load_model_module("graph_gte.py", "graph_gte_rope_test")
    query = tlx.convert_to_tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    key = tlx.convert_to_tensor([[[[5.0, 6.0, 7.0, 8.0]]]])
    cosine = tlx.convert_to_tensor([[[[0.0, 0.0, 0.0, 0.0]]]])
    sine = tlx.convert_to_tensor([[[[1.0, 1.0, 1.0, 1.0]]]])

    rotated_query, rotated_key = graph_gte.apply_rotary_pos_emb(
        query, key, cosine, sine)

    assert torch.equal(
        rotated_query,
        torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]]),
    )
    assert torch.equal(
        rotated_key,
        torch.tensor([[[[-7.0, -8.0, 5.0, 6.0]]]]),
    )


def test_graph_gte_has_rope_without_absolute_position_embeddings():
    _patch_tlx_torch_named_members()
    _load_model_module("graph_bert.py", "graph_bert")
    GraphGTE = _load_model_module(
        "graph_gte.py", "graph_gte_position_test").GraphGTE
    model = GraphGTE(
        vocab_size=32,
        output_dim=2,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
    )

    assert not hasattr(model, "position_embeddings")
    assert model.manifest()["position_embedding_type"] == "rope"


def test_graph_gte_manifest_reports_strict_paper_defaults():
    _patch_tlx_torch_named_members()
    _load_model_module("graph_bert.py", "graph_bert")
    GraphGTE = _load_model_module(
        "graph_gte.py", "graph_gte_manifest_test").GraphGTE
    model = GraphGTE(vocab_size=64, output_dim=1)

    manifest = model.manifest()

    assert manifest["encoder_type"] == "gte"
    assert manifest["model_name"] == "gte-base"
    assert manifest["weight_source"] == "random_initialization"
    assert manifest["official_checkpoint"] is None
    assert manifest["hidden_size"] == 768
    assert manifest["num_hidden_layers"] == 12
    assert manifest["num_attention_heads"] == 12
    assert manifest["intermediate_size"] == 3072
    assert manifest["hidden_act"] == "gelu"
    assert manifest["hidden_dropout_prob"] == 0.1
    assert manifest["attention_probs_dropout_prob"] == 0.1
    assert manifest["position_embedding_type"] == "rope"
    assert manifest["max_position_embeddings"] == 8192
    assert manifest["layer_norm_eps"] == 1e-12
    assert manifest["rope_theta"] == 20000.0
    assert manifest["rope_scaling"] == {"type": "ntk", "factor": 8.0}
    assert 110_000_000 < manifest["total_parameters"] < 125_000_000
    assert manifest["trainable_parameters"] == manifest["total_parameters"]


def test_graph_gte_supports_paper_outputs_padding_and_gradients():
    _patch_tlx_torch_named_members()
    _load_model_module("graph_bert.py", "graph_bert")
    GraphGTE = _load_model_module(
        "graph_gte.py", "graph_gte_training_test").GraphGTE
    model = GraphGTE(
        vocab_size=32,
        output_dim=2,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
        dropout_rate=0.0,
    )
    attention_mask = tlx.convert_to_tensor([[1, 1, 1, 0]], dtype=tlx.int64)
    input_ids = tlx.convert_to_tensor([[3, 5, 4, 0]], dtype=tlx.int64)

    mlm_logits = model(input_ids, attention_mask=attention_mask, task="mlm")
    task_logits = model(
        input_ids, attention_mask=attention_mask, task="supervised")

    assert tlx.get_tensor_shape(mlm_logits) == [1, 4, 32]
    assert tlx.get_tensor_shape(task_logits) == [1, 2]
    loss = mlm_logits.sum() + task_logits.sum()
    loss.backward()
    checked = [
        model.encoder_layers[0].attention.qkv_proj.trainable_weights[0],
        model.encoder_layers[0].mlp.up_gate_proj.trainable_weights[0],
        model.mlm_head.trainable_weights[0],
        model.task_head_out.trainable_weights[0],
    ]
    for parameter in checked:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0

    model.set_eval()
    changed_padding = tlx.convert_to_tensor([[3, 5, 4, 17]], dtype=tlx.int64)
    first = model(input_ids, attention_mask=attention_mask, task="supervised")
    second = model(
        changed_padding, attention_mask=attention_mask, task="supervised")
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_graph_tokenizer_trainer_resolves_paper_dataset_aliases():
    trainer = _load_trainer_module()

    qm9 = trainer.resolve_dataset_spec("QM9")
    molhiv = trainer.resolve_dataset_spec("OGBG-molhiv")
    peptides = trainer.resolve_dataset_spec("p-func")
    peptides_struct = trainer.resolve_dataset_spec("Peptides-struct")

    assert qm9.canonical_name == "qm9"
    assert qm9.task_type == "regression"
    assert qm9.output_dim == 16
    assert qm9.metric == "mae"
    assert molhiv.canonical_name == "molhiv"
    assert molhiv.output_dim == 1
    assert molhiv.metric == "rocauc"
    assert peptides.canonical_name == "peptides-func"
    assert peptides.output_dim == 10
    assert peptides.metric == "ap"
    assert peptides_struct.canonical_name == "peptides-struct"
    assert peptides_struct.task_type == "multi_target_regression"
    assert peptides_struct.output_dim == 11
    assert peptides_struct.metric == "average_mae"


def test_graph_tokenizer_trainer_pads_and_masks_token_batches():
    trainer = _load_trainer_module()
    GraphTokenizer, _ = trainer.load_tokenizer_classes()
    tokenizer = GraphTokenizer()

    batch = tokenizer.build_mlm_batch_from_token_sequences(
        [[3, 6, 4], [3, 8, 9, 4]], max_length=5, mask_prob=1.0, seed=7)

    assert batch.input_ids == [[3, 2, 4, 0, 0], [3, 2, 2, 4, 0]]
    assert batch.attention_mask == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]
    assert batch.labels == [[-100, 6, -100, -100, -100], [-100, 8, 9, -100, -100]]
    assert not hasattr(trainer, "pad_token_sequences")
    assert not hasattr(trainer, "apply_mlm_mask")


def test_graph_tokenizer_trainer_builds_mlm_pretrain_batches():
    trainer = _load_trainer_module()

    encoded_split = {
        "input_ids": [[3, 6, 4, 0], [3, 7, 8, 4]],
        "attention_mask": [[1, 1, 1, 0], [1, 1, 1, 1]],
        "labels": [[0.1], [0.2]],
    }
    GraphTokenizer, _ = trainer.load_tokenizer_classes()
    batch = trainer.build_mlm_pretrain_split(
        encoded_split,
        tokenizer=GraphTokenizer(),
        mask_prob=1.0,
        seed=9,
    )

    assert batch == {
        "input_ids": [[3, 2, 4, 0], [3, 7, 2, 4]],
        "attention_mask": [[1, 1, 1, 0], [1, 1, 1, 1]],
        "mlm_labels": [[-100, 6, -100, -100], [-100, -100, 8, -100]],
    }
    assert trainer.count_mlm_labels(batch["mlm_labels"]) == 2


def test_graph_tokenizer_trainer_loads_preprocessed_split_files():
    trainer = _load_trainer_module()
    spec = trainer.resolve_dataset_spec("qm9")

    graphs = [
        types.SimpleNamespace(edge_index=[[0], [1]], x=[index, index + 1], edge_attr=[index], y=[float(index)] * 16)
        for index in range(3)
    ]
    class Dataset:
        def __len__(self):
            return len(graphs)

        def __getitem__(self, index):
            return graphs[index]

        def get_idx_split(self):
            return {"train": [0, 2], "val": [1], "test": []}

    dataset = Dataset()
    trainer.load_gammagl_benchmark_dataset = lambda _root, _spec: dataset

    splits = trainer.load_benchmark_splits("unused", spec)

    assert [len(splits[name]) for name in ("train", "val", "test")] == [2, 1, 0]
    assert splits["train"][0].edge_index == [[0], [1]]
    assert splits["train"][0].x == [0, 1]
    assert splits["train"][0].edge_attr == [0]
    assert splits["train"][0].y[:2] == [0.0, 0.0]
    assert [graph.graph_tokenizer_id for graph in splits["train"]] == [0, 2]


def test_graph_tokenizer_dataset_loader_bootstraps_repo_import_path(tmp_path, monkeypatch):
    trainer = _load_trainer_module()
    spec = trainer.resolve_dataset_spec("qm9")
    calls = []
    fake_datasets = types.SimpleNamespace(QM9=lambda root: {"root": root})
    fake_gammagl = types.ModuleType("gammagl")
    fake_gammagl.datasets = fake_datasets

    monkeypatch.setattr(trainer, "ensure_repo_on_path", lambda: calls.append(True))
    monkeypatch.setitem(sys.modules, "gammagl", fake_gammagl)

    dataset = trainer.load_gammagl_benchmark_dataset(tmp_path, spec)

    assert dataset == {"root": str(tmp_path)}
    assert calls == [True]


def test_graph_tokenizer_bpe_preflight_bootstraps_repo_import_path(monkeypatch):
    trainer = _load_trainer_module()
    calls = []
    fake_third_party = types.ModuleType("third_party")
    fake_third_party.graph_bpe_cpp = types.SimpleNamespace(is_available=lambda: True)

    monkeypatch.setattr(trainer, "ensure_repo_on_path", lambda: calls.append(True))
    monkeypatch.setitem(sys.modules, "third_party", fake_third_party)

    status = trainer.preflight_bpe_backend_status("cpp")

    assert status == {"backend": "cpp", "status": "ok", "native_available": True}
    assert calls == [True]


def test_graph_tokenizer_trainer_rejects_benchmarks_without_a_public_dataset():
    trainer = _load_trainer_module()
    with pytest.raises(ValueError, match="QM9"):
        trainer.load_gammagl_benchmark_dataset("unused", trainer.resolve_dataset_spec("p-func"))


def test_graph_tokenizer_trainer_computes_primary_metrics():
    trainer = _load_trainer_module()

    qm9 = trainer.resolve_dataset_spec("qm9")
    molhiv = trainer.resolve_dataset_spec("molhiv")
    peptides = trainer.resolve_dataset_spec("p-func")
    peptides_struct = trainer.resolve_dataset_spec("p-struct")

    assert trainer.compute_primary_metric(qm9, [[1.0, 2.0]], [[2.0, 4.0]]) == 1.5
    assert trainer.compute_primary_metric(molhiv, [0, 1, 0, 1], [0.1, 0.9, 0.4, 0.8]) == 1.0
    assert trainer.compute_primary_metric(peptides, [[1, 0], [0, 1]], [[0.9, 0.1], [0.2, 0.8]]) == 1.0
    assert trainer.compute_primary_metric(
        peptides_struct,
        [[1.0, 2.0], [3.0, float("nan")]],
        [[2.0, 4.0], [1.0, 100.0]],
    ) == 1.75


def test_shared_mlm_helper_forces_one_mask_when_random_draw_selects_none():
    GraphTokenizer, _ = _load_trainer_module().load_tokenizer_classes()
    masked, labels = GraphTokenizer().mask_token_sequences(
        [[3, 11, 12, 4]], mask_prob=1e-12, seed=0)

    selected = [
        index for index, label in enumerate(labels[0]) if label != -100
    ]
    assert len(selected) == 1
    assert labels[0][selected[0]] in {11, 12}
    assert masked[0][selected[0]] == 2


def test_graph_tokenizer_trainer_reports_task_metrics_by_dataset():
    trainer = _load_trainer_module()

    qm9 = trainer.resolve_dataset_spec("qm9")
    molhiv = trainer.resolve_dataset_spec("molhiv")
    peptides_func = trainer.resolve_dataset_spec("p-func")
    peptides_struct = trainer.resolve_dataset_spec("p-struct")

    qm9_metrics = trainer.compute_task_metrics(qm9, [[1.0, 2.0]], [[2.0, 4.0]])
    molhiv_metrics = trainer.compute_task_metrics(molhiv, [0, 1, 0, 1], [0.1, 0.9, 0.4, 0.8])
    peptides_func_metrics = trainer.compute_task_metrics(
        peptides_func,
        [[1, 0], [0, 1]],
        [[0.9, 0.1], [0.2, 0.8]],
    )
    peptides_struct_metrics = trainer.compute_task_metrics(
        peptides_struct,
        [[1.0, 2.0], [3.0, float("nan")]],
        [[2.0, 4.0], [1.0, 100.0]],
    )

    assert qm9_metrics == {"mae": 1.5}
    assert molhiv_metrics == {"auc": 1.0}
    assert peptides_func_metrics == {"ap": 1.0}
    assert peptides_struct_metrics == {"average_mae": 1.75, "mae": 1.6666666666666667}
    assert trainer.accuracy_score([0, 1, 1, 0], [0.2, 0.8, 0.7, 0.4]) == 1.0


def test_graph_tokenizer_trainer_selects_best_epoch_by_metric_direction():
    trainer = _load_trainer_module()

    qm9 = trainer.resolve_dataset_spec("qm9")
    molhiv = trainer.resolve_dataset_spec("molhiv")

    regression_history = [
        {"epoch": 1, "val": {"metrics": {"mae": 0.9}}, "test": {"metrics": {"mae": 1.1}}},
        {"epoch": 2, "val": {"metrics": {"mae": 0.7}}, "test": {"metrics": {"mae": 1.0}}},
        {"epoch": 3, "val": {"metrics": {"mae": 0.8}}, "test": {"metrics": {"mae": 0.6}}},
    ]
    classification_history = [
        {"epoch": 1, "val": {"metrics": {"auc": 0.71}}, "test": {"metrics": {"auc": 0.69}}},
        {"epoch": 2, "val": {"metrics": {"auc": 0.68}}, "test": {"metrics": {"auc": 0.80}}},
        {"epoch": 3, "val": {"metrics": {"auc": 0.74}}, "test": {"metrics": {"auc": 0.73}}},
    ]

    assert trainer.select_best_epoch(qm9, regression_history)["epoch"] == 2
    assert trainer.select_best_epoch(molhiv, classification_history)["epoch"] == 3


def test_graph_tokenizer_trainer_evaluates_test_once_after_restoring_best_val(monkeypatch, tmp_path):
    trainer = _load_trainer_module()
    spec = trainer.resolve_dataset_spec("qm9")
    splits = {"train": [object()], "val": [object()], "test": [object()]}
    encoded = {name: {"split": name} for name in splits}
    calls = []

    class Model:
        trainable_weights = []

        def __init__(self, **_kwargs):
            self.loaded_state = None

        def state_dict(self):
            return {"epoch": len(calls)}

        def load_state_dict(self, state):
            self.loaded_state = state

    model = Model()
    fake_tlx = types.ModuleType("tensorlayerx")
    fake_tlx.optimizers = SimpleNamespace(Adam=lambda **_kwargs: object())
    fake_tlx_model = types.ModuleType("tensorlayerx.model")
    fake_tlx_model.TrainOneStep = lambda *_args: object()
    monkeypatch.setitem(sys.modules, "tensorlayerx", fake_tlx)
    monkeypatch.setitem(sys.modules, "tensorlayerx.model", fake_tlx_model)
    monkeypatch.setattr(trainer, "load_benchmark_splits", lambda *_args: splits)
    monkeypatch.setattr(trainer, "fit_tokenizer", lambda *_args: object())
    monkeypatch.setattr(trainer, "encode_graph_splits", lambda *_args, **_kwargs: encoded)
    monkeypatch.setattr(trainer, "patch_tlx_torch_named_members", lambda: None)
    monkeypatch.setattr(trainer, "load_model_class", lambda _name: lambda **_kwargs: model)
    monkeypatch.setattr(trainer, "model_kwargs", lambda *_args: {})
    monkeypatch.setattr(trainer, "run_mlm_pretrain", lambda *_args: {})
    monkeypatch.setattr(trainer, "GraphTokenSupervisedLoss", lambda *_args: object())
    monkeypatch.setattr(trainer, "train_one_epoch", lambda *_args: 0.0)

    val_metrics = iter((0.4, 0.2, 0.3))

    def evaluate(_model, encoded_split, _spec, _args):
        split = encoded_split["split"]
        calls.append(split)
        metric = next(val_metrics) if split == "val" else 0.1
        return {"loss": metric, "metrics": {"mae": metric}}

    monkeypatch.setattr(trainer, "evaluate_split", evaluate)
    args = SimpleNamespace(
        dataset="qm9", seed=0, data_root=tmp_path, max_length=8, model="bert",
        pretrain_epoch=0, lr=0.001, weight_decay=0.0, n_epoch=3, patience=0,
        batch_size=1, save_checkpoint=False, run=0, output_dir=tmp_path,
    )

    summary = trainer.run_train_val_test(args)

    assert calls == ["val", "val", "val", "test"]
    assert model.loaded_state == {"epoch": 2}
    assert all("test" not in entry for entry in summary["history"])
    assert summary["best_test"]["metrics"] == {"mae": 0.1}


def test_graph_tokenizer_trainer_iterates_token_batches_with_labels():
    trainer = _load_trainer_module()

    batches = list(
        trainer.iter_token_batches(
            {
                "input_ids": [[1, 2], [3, 4], [5, 6]],
                "attention_mask": [[1, 1], [1, 1], [1, 0]],
                "labels": [[0.1], [0.2], [0.3]],
            },
            batch_size=2,
        )
    )

    assert batches == [
        {
            "input_ids": [[1, 2], [3, 4]],
            "attention_mask": [[1, 1], [1, 1]],
            "labels": [[0.1], [0.2]],
        },
        {
            "input_ids": [[5, 6]],
            "attention_mask": [[1, 0]],
            "labels": [[0.3]],
        },
    ]


def test_graph_tokenizer_trainer_rejects_multidimensional_features_without_adapter():
    trainer = _load_trainer_module()

    with pytest.raises(ValueError, match="dataset-specific token adapter"):
        trainer.flatten_feature_ids([[1, 2], [1, 9]])


def test_graph_tokenizer_trainer_formats_result_rows_and_aggregates_runs():
    trainer = _load_trainer_module()
    spec = trainer.resolve_dataset_spec("molhiv")
    summaries = [
        {
            "dataset": "molhiv",
            "model": "bert",
            "run": 0,
            "seed": 7,
            "primary_metric": "auc",
            "best_epoch": 2,
            "best_val": {"loss": 0.4, "metrics": {"auc": 0.8}},
            "best_test": {"loss": 0.5, "metrics": {"auc": 0.7}},
        },
        {
            "dataset": "molhiv",
            "model": "bert",
            "run": 1,
            "seed": 8,
            "primary_metric": "auc",
            "best_epoch": 3,
            "best_val": {"loss": 0.3, "metrics": {"auc": 0.9}},
            "best_test": {"loss": 0.6, "metrics": {"auc": 0.9}},
        },
    ]

    assert trainer.format_result_row(summaries[0]) == {
        "dataset": "molhiv",
        "model": "bert",
        "run": 0,
        "seed": 7,
        "best_epoch": 2,
        "val_auc": 0.8,
        "test_auc": 0.7,
        "val_loss": 0.4,
        "test_loss": 0.5,
    }
    assert trainer.aggregate_run_summaries(spec, summaries) == {
        "num_runs": 2,
        "primary_metric": "auc",
        "higher_is_better": True,
        "val_auc_mean": 0.8500000000000001,
        "val_auc_std": 0.04999999999999999,
        "test_auc_mean": 0.8,
        "test_auc_std": 0.10000000000000003,
    }


def test_graph_tokenizer_aggregates_per_target_mae_across_runs():
    trainer = _load_trainer_module()
    runs = [
        {
            "best_val": {"per_target_mae": {"mu": 1.0, "alpha": 3.0}},
            "best_test": {"per_target_mae": {"mu": 2.0, "alpha": 4.0}},
        },
        {
            "best_val": {"per_target_mae": {"mu": 3.0, "alpha": 5.0}},
            "best_test": {"per_target_mae": {"mu": 4.0, "alpha": 6.0}},
        },
    ]

    aggregate = trainer.aggregate_per_target_mae(runs)

    assert aggregate == {
        "val": {
            "mu": {"mean": 2.0, "std": 1.0},
            "alpha": {"mean": 4.0, "std": 1.0},
        },
        "test": {
            "mu": {"mean": 3.0, "std": 1.0},
            "alpha": {"mean": 5.0, "std": 1.0},
        },
    }


def test_graph_tokenizer_aggregates_raw_per_target_mae_across_runs():
    trainer = _load_trainer_module()
    runs = [
        {
            "best_val": {"per_target_mae_raw": {"mu": 10.0}},
            "best_test": {"per_target_mae_raw": {"mu": 20.0}},
        },
        {
            "best_val": {"per_target_mae_raw": {"mu": 30.0}},
            "best_test": {"per_target_mae_raw": {"mu": 40.0}},
        },
    ]

    assert trainer.aggregate_per_target_mae(
        runs, field_name="per_target_mae_raw") == {
            "val": {"mu": {"mean": 20.0, "std": 10.0}},
            "test": {"mu": {"mean": 30.0, "std": 10.0}},
        }


def test_graph_tokenizer_trainer_formats_paper_result_tables():
    trainer = _load_trainer_module()

    aggregate = {
        "num_runs": 3,
        "primary_metric": "auc",
        "higher_is_better": True,
        "val_auc_mean": 0.81234,
        "val_auc_std": 0.01234,
        "test_auc_mean": 0.76543,
        "test_auc_std": 0.02345,
    }
    summary = {
        "dataset": "molhiv",
        "model": "bert",
        "aggregate": aggregate,
    }

    table_row = trainer.format_paper_table_row(summary)

    assert table_row == {
        "dataset": "molhiv",
        "model": "bert",
        "metric": "auc",
        "direction": "higher",
        "runs": 3,
        "val": "0.8123 +/- 0.0123",
        "test": "0.7654 +/- 0.0234",
    }
    assert trainer.format_markdown_result_table([summary]).splitlines() == [
        "| Dataset | Model | Metric | Direction | Runs | Val | Test |",
        "|---|---|---|---|---:|---:|---:|",
        "| molhiv | bert | auc | higher | 3 | 0.8123 +/- 0.0123 | 0.7654 +/- 0.0234 |",
    ]
    assert trainer.format_latex_result_rows([summary]) == (
        "molhiv & bert & auc & higher & 3 & 0.8123 $\\pm$ 0.0123 & 0.7654 $\\pm$ 0.0234 \\"
    )


def test_graph_tokenizer_trainer_parses_experiment_matrix():
    trainer = _load_trainer_module()

    assert trainer.parse_csv_values("qm9, molhiv,,p-func") == ["qm9", "molhiv", "p-func"]

    class Args:
        dataset = "qm9"
        datasets = "qm9,p-struct"
        model = "bert"
        models = "bert,gte"

    assert trainer.experiment_matrix_items(Args()) == [
        ("qm9", "bert"),
        ("qm9", "gte"),
        ("p-struct", "bert"),
        ("p-struct", "gte"),
    ]


def test_graph_tokenizer_trainer_formats_combined_paper_tables():
    trainer = _load_trainer_module()

    summaries = [
        {
            "dataset": "qm9",
            "model": "bert",
            "aggregate": {
                "num_runs": 2,
                "primary_metric": "mae",
                "higher_is_better": False,
                "val_mae_mean": 0.21,
                "val_mae_std": 0.01,
                "test_mae_mean": 0.23,
                "test_mae_std": 0.02,
            },
        },
        {
            "dataset": "molhiv",
            "model": "gte",
            "aggregate": {
                "num_runs": 2,
                "primary_metric": "auc",
                "higher_is_better": True,
                "val_auc_mean": 0.81,
                "val_auc_std": 0.03,
                "test_auc_mean": 0.78,
                "test_auc_std": 0.04,
            },
        },
    ]

    markdown = trainer.format_markdown_result_table(summaries)
    latex = trainer.format_latex_result_rows(summaries)

    assert "| qm9 | bert | mae | lower | 2 | 0.2100 +/- 0.0100 | 0.2300 +/- 0.0200 |" in markdown
    assert "| molhiv | gte | auc | higher | 2 | 0.8100 +/- 0.0300 | 0.7800 +/- 0.0400 |" in markdown
    assert "qm9 & bert & mae & lower & 2 & 0.2100 $\\pm$ 0.0100 & 0.2300 $\\pm$ 0.0200" in latex
    assert "molhiv & gte & auc & higher & 2 & 0.8100 $\\pm$ 0.0300 & 0.7800 $\\pm$ 0.0400" in latex


def test_graph_tokenizer_trainer_loads_existing_experiment_for_resume(tmp_path):
    trainer = _load_trainer_module()

    class Args:
        output_dir = str(tmp_path)

    summary_path = tmp_path / "qm9_bert_summary.json"
    summary_path.write_text(
        json.dumps({"dataset": "qm9", "model": "bert", "aggregate": {"num_runs": 1}}),
        encoding="utf-8",
    )

    loaded = trainer.load_existing_experiment_summary(Args(), "qm9", "bert")

    assert loaded["dataset"] == "qm9"
    assert loaded["model"] == "bert"
    assert loaded["status"] == "skipped_existing"
    assert loaded["outputs"]["json"] == str(summary_path)


def test_graph_tokenizer_trainer_records_matrix_failures():
    trainer = _load_trainer_module()

    failure = trainer.format_failed_experiment_summary("qm9", "bert", RuntimeError("boom"))

    assert failure["dataset"] == "qm9"
    assert failure["model"] == "bert"
    assert failure["status"] == "failed"
    assert failure["error_type"] == "RuntimeError"
    assert failure["error"] == "boom"
    assert failure["aggregate"]["num_runs"] == 0


def test_graph_tokenizer_trainer_serializes_args_for_manifest(tmp_path):
    trainer = _load_trainer_module()

    class Args:
        dataset = "qm9"
        model = "bert"
        output_dir = tmp_path
        hidden_size = None
        runs = 3
        values = ("a", "b")

    serialized = trainer.serialize_args(Args())

    assert serialized["dataset"] == "qm9"
    assert serialized["model"] == "bert"
    assert serialized["output_dir"] == str(tmp_path)
    assert serialized["hidden_size"] is None
    assert serialized["runs"] == 3
    assert serialized["values"] == ["a", "b"]


def test_graph_tokenizer_trainer_builds_runtime_manifest():
    trainer = _load_trainer_module()

    class Args:
        dataset = "molhiv"
        model = "gte"
        runs = 2
        output_dir = "logs"

    manifest = trainer.build_runtime_manifest(Args(), timestamp="2026-07-03T00:00:00+00:00")

    assert manifest["timestamp"] == "2026-07-03T00:00:00+00:00"
    assert manifest["args"]["dataset"] == "molhiv"
    assert manifest["args"]["model"] == "gte"
    assert "version" in manifest["python"]
    assert "executable" in manifest["python"]
    assert "platform" in manifest["python"]
    assert "packages" in manifest
    assert "git" in manifest


def test_runtime_manifest_does_not_execute_git_commands():
    trainer = _load_trainer_module()

    assert trainer.git_metadata() == {
        "status": "disabled",
        "reason": "local-only run; git commands are disabled",
    }


def test_graph_tokenizer_trainer_writes_manifest_output(tmp_path):
    trainer = _load_trainer_module()

    class Args:
        output_dir = str(tmp_path)

    summary = {
        "dataset": "qm9",
        "model": "bert",
        "rows": [],
        "aggregate": {
            "num_runs": 1,
            "primary_metric": "mae",
            "higher_is_better": False,
            "val_mae_mean": 0.2,
            "val_mae_std": 0.0,
            "test_mae_mean": 0.3,
            "test_mae_std": 0.0,
        },
        "manifest": {"timestamp": "2026-07-03T00:00:00+00:00", "args": {"dataset": "qm9"}},
    }

    outputs = trainer.write_experiment_outputs(summary, Args())
    manifest_path = Path(outputs["manifest"])

    assert manifest_path.name == "qm9_bert_manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["timestamp"] == "2026-07-03T00:00:00+00:00"



def test_graph_tokenizer_trainer_parses_paper_protocol_options():
    trainer = _load_trainer_module()

    args = trainer.parse_args([
        "--protocol", "paper",
        "--dataset", "qm9",
        "--model", "gte",
        "--target-property", "homo",
        "--device", "cuda:0",
        "--pretrain-epoch", "200",
        "--n-epoch", "200",
        "--pretrain-lr", "0.00005",
        "--lr", "0.00001",
        "--batch-size", "32",
        "--weight-decay", "0.1",
        "--pretrain-warmup-ratio", "0.12",
        "--finetune-warmup-ratio", "0.025",
        "--pretrain-max-grad-norm", "2.0",
        "--finetune-max-grad-norm", "0.5",
        "--mask-prob", "0.09",
        "--patience", "20",
        "--max-position-embeddings", "8192",
        "--paper-amp", "bf16",
        "--paper-tf32",
        "--paper-loss", "huber",
        "--molhiv-pos-weight",
    ])

    assert args.protocol == "paper"
    assert args.target_property == "homo"
    assert args.device == "cuda:0"
    assert args.pretrain_epoch == 200
    assert args.n_epoch == 200
    assert args.pretrain_lr == 5e-5
    assert args.lr == 1e-5
    assert args.batch_size == 32
    assert args.weight_decay == 0.1
    assert args.pretrain_warmup_ratio == 0.12
    assert args.finetune_warmup_ratio == 0.025
    assert args.pretrain_max_grad_norm == 2.0
    assert args.finetune_max_grad_norm == 0.5
    assert args.mask_prob == 0.09
    assert args.patience == 20
    assert args.max_position_embeddings == 8192
    assert args.paper_amp == "bf16"
    assert args.paper_tf32 is True
    assert args.paper_loss == "huber"
    assert args.molhiv_pos_weight is True



def test_graph_tokenizer_trainer_rejects_unknown_model_name():
    trainer = _load_trainer_module()

    try:
        trainer.resolve_model_name("bad-model")
    except ValueError as error:
        assert "Unsupported model" in str(error)
    else:
        raise AssertionError("unknown model name was accepted")


def test_graph_tokenizer_trainer_rejects_overlapping_dataset_splits(monkeypatch):
    trainer = _load_trainer_module()
    class Dataset(list):
        def get_idx_split(self):
            return {"train": [0], "val": [0], "test": [1]}

    dataset = Dataset([
        types.SimpleNamespace(edge_index=[[], []], x=[], edge_attr=[], y=[0.0] * 16)
        for _ in range(2)
    ])
    monkeypatch.setattr(trainer, "load_gammagl_benchmark_dataset", lambda *_args: dataset)

    with pytest.raises(ValueError, match="overlaps"):
        trainer.load_benchmark_splits("unused", trainer.resolve_dataset_spec("qm9"))


def test_paper_tokenizer_reuses_train_only_local_cache(tmp_path):
    trainer = _load_trainer_module()
    args = types.SimpleNamespace(
        protocol="paper",
        dataset="qm9",
        data_root=str(tmp_path / "data"),
        paper_cache_root=str(tmp_path / "cache"),
        serialization="feuler",
        num_merges=1,
        min_frequency=1,
        bpe_backend="python",
    )
    graphs = [
        trainer.SyntheticGraph(
            edge_index=[[0], [1]], x=[13, 15], edge_attr=[2], y=[0.0]),
        trainer.SyntheticGraph(
            edge_index=[[0], [1]], x=[13, 17], edge_attr=[2], y=[1.0]),
    ]

    first = trainer.fit_tokenizer(args, graphs)
    second = trainer.fit_tokenizer(args, graphs)

    assert first._cache_status == "miss"
    assert second._cache_status == "hit"
    assert second.bpe.codebook.merge_rules == first.bpe.codebook.merge_rules
    assert len(list((tmp_path / "cache").rglob("tokenizer.pkl"))) == 1


def test_graph_tokenizer_trainer_preflight_reports_missing_dataset(tmp_path, monkeypatch):
    trainer = _load_trainer_module()
    monkeypatch.setattr(
        trainer, "load_gammagl_benchmark_dataset", lambda data_root, spec: None)

    class Args:
        data_root = str(tmp_path)
        dataset = "qm9"
        datasets = "qm9"
        model = "bert"
        models = "bert"
        bpe_backend = "python"

    report = trainer.preflight_check(Args())

    assert report["status"] == "failed"
    assert report["num_errors"] == 1
    assert report["datasets"][0]["status"] == "failed"
    assert "dataset loader could not materialize" in report["errors"][0]
