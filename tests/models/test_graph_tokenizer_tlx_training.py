"""Real TLX training checks, runnable in separate torch and TF eager processes."""
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorlayerx as tlx

pytestmark = pytest.mark.skipif(tlx.BACKEND not in ("torch", "tensorflow"),
                                reason="GraphTokenizer training supports torch and TensorFlow eager")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def trainer():
    spec = importlib.util.spec_from_file_location(
        "graph_tokenizer_tlx_trainer", ROOT / "examples/graph_tokenizer/graph_tokenizer_trainer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ScalarModel(tlx.nn.Module):
    def __init__(self, fault=None):
        super().__init__()
        self.weight = self._get_weights("weight", shape=(1,), init=tlx.initializers.Constant(1.0))
        self.fault = fault

    def forward(self, inputs, attention=None, task=None):
        value = self.weight
        if self.fault == "gradient":
            value = tlx.sqrt(value - 1.0) + 1.0
        if self.fault == "logits":
            value = value * float("nan")
        values = tlx.cast(inputs, tlx.float32) * value
        if task == "mlm":
            return tlx.stack((values, -values, values * 0), axis=-1)
        return values


def batches():
    return [(tlx.convert_to_tensor(x, dtype=tlx.int64), tlx.ones((len(x), 1), dtype=tlx.int64),
             tlx.zeros((len(x), 1)), tlx.convert_to_tensor(list(range(len(x))), dtype=tlx.int64))
            for x in ([[1], [1]], [[3]])]


def args(**overrides):
    return SimpleNamespace(**{**dict(
        finetune_noise_probability=0.0, finetune_noise_std=0.0, finetune_max_grad_norm=1e6,
        pretrain_max_grad_norm=1e6, gradient_accumulation_steps=2, mask_probability=0.0), **overrides})


def test_tlx_optimizer_normalizes_unequal_and_tail_windows(trainer):
    model = ScalarModel()
    optimizer = tlx.optimizers.Adam(lr=0.0)
    steps = []
    original = optimizer.apply_gradients
    observed = []

    def apply(pairs):
        pairs = list(pairs)
        observed.append(float(tlx.convert_to_numpy(pairs[0][0]).reshape(-1)[0]))
        return original(pairs)

    optimizer.apply_gradients = apply
    loss = trainer.train_downstream_epoch(
        tlx, model, batches(), optimizer, SimpleNamespace(step=lambda: steps.append(1)),
        trainer.resolve_dataset("aqsol"), args(), "cpu", seed=7, epoch=1)
    assert loss == pytest.approx(11 / 3)
    assert observed == pytest.approx([22 / 3])
    assert steps == [1]
    # A full two-microbatch window followed by a one-microbatch tail.
    observed.clear()
    trainer.train_downstream_epoch(
        tlx, model, batches() + batches()[:1], optimizer, SimpleNamespace(step=lambda: steps.append(1)),
        trainer.resolve_dataset("aqsol"), args(), "cpu", seed=7, epoch=2)
    assert observed == pytest.approx([22 / 3, 2.0])
    assert steps == [1, 1, 1]


@pytest.mark.parametrize("fault", ["logits", "gradient", "loss"])
def test_tlx_numerical_failure_prevents_update(trainer, fault):
    model = ScalarModel(fault)
    optimizer = tlx.optimizers.Adam(lr=0.1)
    inputs = batches()
    if fault == "loss":
        first = inputs[0]
        inputs[0] = (*first[:2], tlx.ones((2, 1)) * float("nan"), first[3])
    before = tlx.convert_to_numpy(model.weight).copy()
    with pytest.raises(FloatingPointError, match=rf"finetune.*epoch=1.*batch=.*{fault}"):
        trainer.train_downstream_epoch(
            tlx, model, inputs, optimizer, SimpleNamespace(step=lambda: pytest.fail("scheduler advanced")),
            trainer.resolve_dataset("aqsol"), args(), "cpu", seed=7, epoch=1)
    np.testing.assert_array_equal(tlx.convert_to_numpy(model.weight), before)


def test_tlx_empty_mlm_windows_skip_optimizer_and_scheduler(trainer):
    model = ScalarModel()
    tokenizer = SimpleNamespace(special_tokens=SimpleNamespace(pad_token_id=0, mask_token_id=2),
                                vocabulary={10: 1})
    optimizer = tlx.optimizers.Adam(lr=0.1)
    optimizer.apply_gradients = lambda _pairs: pytest.fail("empty MLM updated optimizer")
    loss = trainer.train_mlm_epoch(
        tlx, model, batches(), optimizer,
        SimpleNamespace(step=lambda: pytest.fail("empty MLM advanced scheduler")), tokenizer,
        args(), "cpu", seed=7, epoch=1)
    assert loss == 0.0
    np.testing.assert_array_equal(tlx.convert_to_numpy(model.weight), [1.0])


@pytest.mark.parametrize("encoder", ["bert", "gte"])
def test_tlx_tiny_native_pipeline(trainer, tmp_path, encoder):
    argv = ["--smoke", "--dataset", "aqsol", "--encoder", encoder, "--device", "cpu",
            "--seeds", "42", "--gradient-accumulation-steps", "2", "--output-dir", str(tmp_path)]
    if encoder == "gte":
        argv += ["--allow-random-gte-init"]
    summary = trainer.main(argv)
    assert summary["complete"] is True
    assert math.isfinite(summary["mean"])
    path = tmp_path / "aqsol" / encoder / "run_01_seed_42" / "summary.json"
    result = json.loads(path.read_text(), parse_constant=lambda s: pytest.fail(s))
    assert Path(result["checkpoints"]["finetune"]).is_file()
    assert result["finetune"]["test"]["metric_space"] == "raw_label"
    if encoder == "gte":
        assert result["gte_initialization"]["reproduction"] is False


@pytest.mark.parametrize("path", ["classification", "mlm", "mlm_empty"])
@pytest.mark.parametrize("max_norm", [1e6, 0.2])
def test_tlx_cross_entropy_accumulation_matches_analytic_target_mean(trainer, monkeypatch, path, max_norm):
    class TokenModel(ScalarModel):
        def forward(self, inputs, attention=None, task=None):
            values = tlx.cast(inputs, tlx.float32) * self.weight
            if path == "classification":
                return tlx.concat((values, -values), axis=-1)
            return tlx.stack((values, -values, values * 0), axis=-1)

    model = TokenModel()
    optimizer = tlx.optimizers.Adam(lr=0.0)
    captured, steps = [], []
    original = optimizer.apply_gradients

    def apply(pairs):
        pairs = list(pairs)
        captured.append(float(tlx.convert_to_numpy(pairs[0][0]).reshape(-1)[0]))
        return original(pairs)

    optimizer.apply_gradients = apply
    if path.startswith("mlm"):
        values = [np.array([[1, 2, 3]]), np.array([[2, 1, 3]])]
        labels = [np.array([[0, -100, -100]]), np.array([[1, 2, 1]])]
        if path == "mlm_empty":
            labels[0][:] = -100
        draws = iter(labels)
        monkeypatch.setattr(trainer, "mask_for_mlm", lambda _b, inputs, *_args:
                            (inputs, tlx.convert_to_tensor(next(draws), dtype=tlx.int64)))
        directions = np.array([1.0, -1.0, 0.0])
    else:
        values = [np.array([[1], [2]]), np.array([[3]])]
        labels = [np.array([[0], [1]]), np.array([[0]])]
        directions = np.array([1.0, -1.0])
    loader = [(tlx.convert_to_tensor(x, dtype=tlx.int64), tlx.ones(x.shape, dtype=tlx.int64),
               tlx.convert_to_tensor(y, dtype=tlx.float32), tlx.zeros((len(x),), dtype=tlx.int64))
              for x, y in zip(values, labels)]
    # Independent scalar softmax derivatives, with no trainer loss/helper reuse.
    loss_sum, gradient_sum, targets = 0.0, 0.0, 0
    for inputs, targets_in_batch in zip(values, labels):
        for x, y in zip(inputs.reshape(-1), targets_in_batch.reshape(-1)):
            if y == -100:
                continue
            logits = x * directions
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            loss_sum -= np.log(probabilities[int(y)])
            gradient_sum += x * (probabilities @ directions - directions[int(y)])
            targets += 1
    scheduler = SimpleNamespace(step=lambda: steps.append(1))
    config = args(finetune_max_grad_norm=max_norm, pretrain_max_grad_norm=max_norm)
    if path.startswith("mlm"):
        loss = trainer.train_mlm_epoch(tlx, model, loader, optimizer, scheduler, None, config, "cpu", 7, 1)
    else:
        loss = trainer.train_downstream_epoch(tlx, model, loader, optimizer, scheduler,
                                              trainer.resolve_dataset("mutagenicity"), config, "cpu", 7, 1)
    gradient = gradient_sum / targets
    clipped = gradient * min(1.0, max_norm / (abs(gradient) + 1e-6))
    assert loss == pytest.approx(loss_sum / targets, rel=2e-6)
    assert captured == pytest.approx([clipped], rel=2e-6)
    assert steps == [1]


def test_tlx_checkpoint_restores_weights_and_rejects_invalid_first_validation(trainer, tmp_path, monkeypatch):
    model = ScalarModel()
    checkpoint = tmp_path / ("best.npz" if tlx.BACKEND == "tensorflow" else "best.pt")
    trainer.save_checkpoint(tlx, checkpoint, model, 2, 0.25)
    helpers = trainer.training_helpers()
    helpers.assign(model.weight, tlx.ones_like(model.weight) * 9)
    restored = trainer.restore_checkpoint(tlx, checkpoint, model, "cpu")
    assert restored["epoch"] == 2 and restored["score"] == 0.25
    np.testing.assert_array_equal(tlx.convert_to_numpy(model.weight), [1.0])
    previous = checkpoint.read_bytes()
    monkeypatch.setattr(trainer, "evaluate_downstream", lambda *_args:
                        {"loss": 0.1, "metric": float("nan")})
    config = args(learning_rate=0.01, weight_decay=0.0, finetune_epochs=1,
                  finetune_warmup_ratio=0.0, early_stopping_patience=1, seed=7)
    with pytest.raises(FloatingPointError, match="validation.*epoch=1.*metric"):
        trainer.run_finetuning(tlx, model, batches(), batches(), batches(), trainer.resolve_dataset("aqsol"),
                               None, config, "cpu", checkpoint)
    assert checkpoint.read_bytes() == previous


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_tlx_evaluation_does_not_turn_nonfinite_logits_into_perfect_zero_class(trainer, bad):
    class BadClassifier(ScalarModel):
        def forward(self, inputs, attention=None, task=None):
            return tlx.concat((tlx.ones((len(inputs), 1)) * bad, tlx.zeros((len(inputs), 1))), axis=-1)

    with pytest.raises(FloatingPointError, match="validation.*epoch=1.*batch=1.*logits"):
        trainer.evaluate_downstream(tlx, BadClassifier(), batches(), trainer.resolve_dataset("mutagenicity"),
                                    None, "cpu", "validation", 1)


def test_tlx_scheduler_counts_only_real_optimizer_updates(trainer):
    optimizer = trainer.make_optimizer(0.1, 0.0)
    scheduler = trainer.build_warmup_cosine_scheduler(tlx, optimizer, 10, 0.2)
    assert optimizer.lr == pytest.approx(0.05)
    scheduler.step()
    assert optimizer.lr == pytest.approx(0.1)
    scheduler.step()
    assert optimizer.lr == pytest.approx(0.1)


@pytest.mark.parametrize("batch_size", [1, 2, 6])
def test_tlx_classification_averages_probabilities_across_batches(trainer, batch_size):
    class Classifier(tlx.nn.Module):
        def forward(self, inputs, attention=None, task=None):
            table = tlx.convert_to_tensor([[10., 0.], [0., 5.], [0., 2.],
                                           [5., 0.], [0., 2.], [0., 5.]])
            return tlx.gather(table, inputs[:, 1] - 8)

    gids, labels = [8, 3, 8, 12, 8, 3], [1., 1., 1., 0., 1., 1.]
    records = [dict(input_ids=[3, 8 + i, 4], labels=[label], graph_id=gid)
               for i, (gid, label) in enumerate(zip(gids, labels))]
    for phase in ["validation", "test"]:
        result = trainer.evaluate_downstream(tlx, Classifier(),
            trainer.make_loader(tlx, records, batch_size, 0, False, False),
            trainer.resolve_dataset("dblp"), None, "cpu", phase)
        assert result["num_graphs"] == 3
        assert result["accuracy"] == 1.


@pytest.mark.parametrize("phase", ["pretrain", "finetune", "validation"])
def test_tlx_main_numerical_failure_keeps_summary_incomplete(trainer, monkeypatch, tmp_path, phase):
    class PipelineFault(ScalarModel):
        def forward(self, inputs, attention=None, task=None):
            value = self.weight
            if ((phase == "pretrain" and task == "mlm") or
                    (phase == "finetune" and task == "supervised" and self.is_train) or
                    (phase == "validation" and not self.is_train)):
                value = value * float("nan")
            if task == "mlm":
                return tlx.ones((*inputs.shape, 64)) * value
            return tlx.ones((len(inputs), 1)) * value

    monkeypatch.setattr(trainer, "make_model", lambda *_args: PipelineFault())
    with pytest.raises(FloatingPointError, match=rf"{phase}.*epoch=1.*batch=1.*logits"):
        trainer.main(["--smoke", "--dataset", "aqsol", "--seeds", "42", "--output-dir", str(tmp_path)])
    directory = tmp_path / "aqsol" / "bert"
    partial = json.loads((directory / "summary.json").read_text(), parse_constant=lambda s: pytest.fail(s))
    assert partial["complete"] is False and partial["run_metrics"] == []
    assert not list(directory.rglob("best_finetune.*"))
    assert not (directory / "run_01_seed_42" / "summary.json").exists()
