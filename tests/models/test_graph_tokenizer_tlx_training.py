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


def test_mlm_masking_matches_unmodified_official_draws(trainer):
    from gammagl.transforms import GraphTokenizerSpecialTokens
    fixture = json.loads((Path(__file__).parent / 'fixtures/graph_tokenizer_training_official.json').read_text())['masking']
    tokenizer = SimpleNamespace(special_tokens=GraphTokenizerSpecialTokens(),
                                vocabulary={i: i for i in range(fixture['vocab_size'])})

    class Draws:
        def __init__(self):
            self.draws = iter(fixture['draws'])

        def random(self, shape):
            return np.asarray(next(self.draws))

        def integers(self, low, high, size):
            assert low == 0 and high == fixture['vocab_size']
            return np.asarray(fixture['replacements'])

    masked, labels = trainer.mask_for_mlm(tlx,
        tlx.convert_to_tensor(fixture['input_ids'], dtype=tlx.int64),
        tlx.convert_to_tensor(fixture['attention'], dtype=tlx.int64), tokenizer, 1., Draws())
    np.testing.assert_array_equal(tlx.convert_to_numpy(masked), fixture['masked'])
    np.testing.assert_array_equal(tlx.convert_to_numpy(labels), fixture['labels'])


def test_validation_loader_matches_official_graph_sampling_across_epochs(trainer):
    fixture = json.loads((Path(__file__).parent / 'fixtures/graph_tokenizer_training_official.json').read_text())
    for case in fixture['sampling']:
        loader = trainer.make_loader(tlx, fixture['records'], 2, 0, False, True, seed=case['seed'])
        for expected in case['epochs']:
            actual = [row for batch in loader for row in tlx.convert_to_numpy(batch[0]).tolist()]
            assert actual == expected


def test_native_pipeline_samples_validation_but_keeps_all_test_variants(trainer, monkeypatch, tmp_path):
    original = trainer.evaluate_downstream
    counts = {}

    def evaluate(backend, model, loader, spec, normalizer, device, phase='evaluation', epoch=None):
        counts[phase] = sum(len(batch[0]) for batch in loader)
        return original(backend, model, loader, spec, normalizer, device, phase, epoch)

    monkeypatch.setattr(trainer, 'evaluate_downstream', evaluate)
    trainer.main(['--smoke', '--dataset', 'aqsol', '--seed', '42', '--num-serializations', '3',
                  '--mask-probability', '1', '--output-dir', str(tmp_path)])
    assert counts == {'validation': 1, 'test': 3}
    run = json.loads((tmp_path / 'aqsol/bert/run_01_seed_42/summary.json').read_text())
    assert run['training_protocol']['version'] == 2
    assert run['training_protocol']['validation_sampling'] == 'random_one_per_graph'
    manifest = json.loads((tmp_path / 'aqsol/bert/run_01_seed_42/run_manifest.json').read_text())
    assert manifest['training_protocol'] == run['training_protocol']
    assert Path(run['checkpoints']['pretrain']).is_file()


def test_mlm_restores_best_validation_weights_before_finetuning(trainer, monkeypatch, tmp_path):
    model = ScalarModel()
    epochs, scores = [], iter([2., 1., 3., 4.])

    def train(*_args):
        epochs.append(len(epochs) + 1)
        trainer.training_helpers().assign(model.weight, tlx.ones_like(model.weight) * epochs[-1])
        return .1

    monkeypatch.setattr(trainer, 'train_mlm_epoch', train)
    monkeypatch.setattr(trainer, 'evaluate_mlm', lambda *_a, **_k:
                        dict(loss=next(scores), steps=1, num_masked_tokens=1), raising=False)
    config = args(pretrain_learning_rate=.01, pretrain_weight_decay=0., pretrain_epochs=8,
                  pretrain_warmup_ratio=0., pretrain_early_stopping_patience=2, seed=42)
    checkpoint = tmp_path / ('best_pretrain.npz' if tlx.BACKEND == 'tensorflow' else 'best_pretrain.pt')
    result = trainer.run_mlm(tlx, model, [None], None, config, 'cpu',
                             val_loader=[None], checkpoint=checkpoint)
    assert epochs == [1, 2, 3, 4]
    assert result['best_epoch'] == 2 and result['best_validation_loss'] == 1.
    np.testing.assert_array_equal(tlx.convert_to_numpy(model.weight), [2.])
    assert checkpoint.is_file()


def test_mlm_validation_uses_official_batch_means_and_ignores_empty_batches(trainer, monkeypatch):
    model = ScalarModel()
    values = iter([(2., 1), (12., 3), (0., 0)])
    # Supply controlled loss sums/counts; exercise real evaluation and aggregation.
    def loss(*_args):
        total, count = next(values)
        return tlx.convert_to_tensor(total), count
    monkeypatch.setattr(trainer, 'mlm_loss', loss)
    batch = batches()[0]
    result = trainer.evaluate_mlm(tlx, model, [batch] * 3, None, args(), 'cpu', seed=42, epoch=1)
    assert result == dict(loss=3., steps=2, num_masked_tokens=4)
    assert not model.is_train


@pytest.mark.parametrize('fault', ['empty', 'nonfinite', 'save', 'restore'])
def test_mlm_selection_failure_keeps_main_incomplete(trainer, monkeypatch, tmp_path, fault):
    if fault == 'empty':
        monkeypatch.setattr(trainer, 'evaluate_mlm', lambda *_a, **_k:
                            dict(loss=None, steps=0, num_masked_tokens=0), raising=False)
    elif fault == 'nonfinite':
        monkeypatch.setattr(trainer, 'evaluate_mlm', lambda *_a, **_k:
                            dict(loss=float('nan'), steps=1, num_masked_tokens=1), raising=False)
    else:
        original = trainer.save_checkpoint if fault == 'save' else trainer.restore_checkpoint
        def fail(backend, path, *a, **k):
            if Path(path).stem == 'best_pretrain':
                raise OSError('pretraining checkpoint failure')
            return original(backend, path, *a, **k)
        monkeypatch.setattr(trainer, 'save_checkpoint' if fault == 'save' else 'restore_checkpoint', fail)
    with pytest.raises((RuntimeError, FloatingPointError, OSError)):
        trainer.main(['--smoke', '--dataset', 'aqsol', '--seed', '42', '--mask-probability', '1',
                      '--output-dir', str(tmp_path)])
    directory = tmp_path / 'aqsol/bert'
    assert json.loads((directory / 'summary.json').read_text())['complete'] is False
    assert json.loads((directory / 'summary.json').read_text())['model_protocol_version'] == 2
    assert not list(directory.rglob('best_finetune.*'))
    assert not (directory / 'run_01_seed_42/summary.json').exists()


def test_aggregation_rejects_mixed_training_protocols(trainer):
    common = dict(dataset='aqsol', encoder='bert', metric_name='MAE', metric=.1,
                  tokenizer_fit_scope='all')
    with pytest.raises(ValueError, match='training protocol'):
        trainer.aggregate_run_results([dict(common, seed=42),
            dict(common, seed=43, training_protocol={'version': 2})])


def test_checkpoint_rejects_other_training_protocol_before_assigning(trainer, tmp_path):
    source, target = ScalarModel(), ScalarModel()
    source.training_protocol_version, target.training_protocol_version = 1, 2
    trainer.training_helpers().assign(source.weight, tlx.ones_like(source.weight) * 9.)
    path = tmp_path / ('best.npz' if tlx.BACKEND == 'tensorflow' else 'best.pt')
    trainer.save_checkpoint(tlx, path, source, 1, .1)
    with pytest.raises(ValueError, match='training protocol'):
        trainer.restore_checkpoint(tlx, path, target, 'cpu')
    np.testing.assert_array_equal(tlx.convert_to_numpy(target.weight), [1.])


def test_sampled_validation_selects_the_official_checkpoint(trainer, monkeypatch, tmp_path):
    class Classifier(ScalarModel):
        def forward(self, inputs, attention=None, task=None):
            table = tlx.convert_to_tensor([[10., 0.], [0., 2.], [0., 2.]])
            return tlx.gather(table, inputs[:, 1] - 8) + self.weight * 0.

    model = Classifier()
    epoch = 0
    def train(*_args):
        nonlocal epoch
        epoch += 1
        trainer.training_helpers().assign(model.weight, tlx.ones_like(model.weight) * epoch)
        return .1
    monkeypatch.setattr(trainer, 'train_downstream_epoch', train)
    records = [dict(input_ids=[3, 8 + i, 4], graph_id=0, labels=[1.]) for i in range(3)]
    # The pinned official dataset with seed 1 chooses variants 0, then 2.
    val = trainer.make_loader(tlx, records, 1, 0, False, True, seed=1)
    test = trainer.make_loader(tlx, records, 2, 0, False, False)
    config = args(learning_rate=.01, finetune_weight_decay=0., finetune_epochs=2,
                  finetune_warmup_ratio=0., early_stopping_patience=2, seed=42)
    checkpoint = tmp_path / ('best.npz' if tlx.BACKEND == 'tensorflow' else 'best.pt')
    result = trainer.run_finetuning(tlx, model, [None], val, test,
        trainer.resolve_dataset('dblp'), None, config, 'cpu', checkpoint)
    assert [row['validation']['metric'] for row in result['history']] == [0., 1.]
    assert result['best_epoch'] == 2 and result['test']['metric'] == 1.
    np.testing.assert_array_equal(tlx.convert_to_numpy(model.weight), [2.])


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
@pytest.mark.parametrize("dataset", ["aqsol", "mutagenicity"])
@pytest.mark.parametrize("fit_scope", ["all", "train"])
def test_tlx_tiny_native_pipeline(trainer, tmp_path, encoder, dataset, fit_scope):
    argv = ["--smoke", "--dataset", dataset, "--encoder", encoder, "--device", "cpu",
            "--seeds", "42", "--fit-scope", fit_scope,
            "--gradient-accumulation-steps", "2", "--output-dir", str(tmp_path)]
    if encoder == "gte":
        argv += ["--allow-random-gte-init"]
    if dataset == "mutagenicity":
        argv += ["--finetune-use-consistency-regularization"]
    summary = trainer.main(argv)
    assert summary["complete"] is True
    assert math.isfinite(summary["mean"])
    assert summary["tokenizer_fit_scope"] == fit_scope
    directory = tmp_path / dataset / encoder
    if fit_scope == "train":
        directory /= "train_only"
    path = directory / "run_01_seed_42" / "summary.json"
    result = json.loads(path.read_text(), parse_constant=lambda s: pytest.fail(s))
    assert result["tokenizer_fit_scope"] == fit_scope
    assert result["tokenizer_fit_split"] == fit_scope
    assert result["normalization_fit_split"] == "train"
    assert Path(result["checkpoints"]["finetune"]).is_file()
    assert result["finetune"]["test"]["metric_space"] == ("raw_label" if dataset == "aqsol" else "class_label")
    assert result["finetune"]["use_consistency_regularization"] is (dataset == "mutagenicity")
    assert result["finetune"]["consistency_alpha"] == 1.0
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


@pytest.mark.parametrize("phase", ["pretrain", "finetune", "validation", "weight"])
@pytest.mark.parametrize("fit_scope", ["all", "train"])
def test_tlx_main_numerical_failure_keeps_summary_incomplete(trainer, monkeypatch, tmp_path, phase, fit_scope):
    from gammagl.models import GraphBERT

    class PipelineFault(GraphBERT):
        def __init__(self):
            super().__init__(64, 1, hidden_size=4, num_hidden_layers=1,
                             num_attention_heads=1, intermediate_size=8, max_position_embeddings=32)
            self.weight = self._get_weights('weight', shape=(1,), init=tlx.initializers.Constant(1.0))
            if phase == "weight":
                self.unused = self._get_weights(
                    "unused", shape=(1,), init=tlx.initializers.Constant(float("nan")))

        def _task_logits(self, pooled):
            return pooled

        def forward(self, inputs, attention=None, task=None):
            value = self.weight
            if ((phase == "pretrain" and task == "mlm") or
                    (phase == "finetune" and task in ("supervised", "pooled") and self.is_train) or
                    (phase == "validation" and not self.is_train)):
                value = value * float("nan")
            if task == "mlm":
                return tlx.ones((*inputs.shape, 64)) * value
            return tlx.ones((len(inputs), 1)) * value

    monkeypatch.setattr(trainer, "make_model", lambda *_args: PipelineFault())
    context = "pretrain.*epoch=1.*batch=1.*weight" if phase == "weight" else rf"{phase}.*epoch=1.*batch=1.*logits"
    with pytest.raises(FloatingPointError, match=context):
        trainer.main(["--smoke", "--dataset", "aqsol", "--seeds", "42",
                      "--fit-scope", fit_scope, "--output-dir", str(tmp_path)])
    directory = tmp_path / "aqsol" / "bert"
    if fit_scope == "train":
        directory /= "train_only"
    partial = json.loads((directory / "summary.json").read_text(), parse_constant=lambda s: pytest.fail(s))
    assert partial["tokenizer_fit_scope"] == fit_scope
    assert partial["complete"] is False and partial["run_metrics"] == []
    assert not list(directory.rglob("best_finetune.*"))
    assert not (directory / "run_01_seed_42" / "summary.json").exists()


def test_tlx_decay_overflow_fails_before_any_update(trainer):
    model = ScalarModel()
    optimizer = trainer.make_optimizer(0.1, 1e308)
    optimizer.apply_gradients = lambda _pairs: pytest.fail("unsafe optimizer update")
    with pytest.raises(FloatingPointError, match="finetune.*weight decay"):
        trainer.train_downstream_epoch(
            tlx, model, batches(), optimizer,
            SimpleNamespace(step=lambda: pytest.fail("scheduler advanced")),
            trainer.resolve_dataset("aqsol"), args(), "cpu", 7, 1)
    np.testing.assert_array_equal(tlx.convert_to_numpy(model.weight), [1.0])


def test_tlx_updated_weight_failure_prevents_scheduler(trainer):
    model = ScalarModel()
    optimizer = trainer.make_optimizer(0.1, 0.0)

    def invalid_update(_pairs):
        trainer.training_helpers().assign(model.weight, model.weight * float("inf"))

    optimizer.apply_gradients = invalid_update
    with pytest.raises(FloatingPointError, match="finetune.*updated weight"):
        trainer.train_downstream_epoch(
            tlx, model, batches(), optimizer,
            SimpleNamespace(step=lambda: pytest.fail("scheduler advanced")),
            trainer.resolve_dataset("aqsol"), args(), "cpu", 7, 1)


def test_tlx_checkpoint_rejects_unused_nonfinite_weight_before_writing(trainer, tmp_path):
    model = ScalarModel()
    model.unused = model._get_weights("unused", shape=(1,), init=tlx.initializers.Constant(1.0))
    checkpoint = tmp_path / "best"
    trainer.save_checkpoint(tlx, checkpoint, model, 1, 0.25)
    previous = checkpoint.read_bytes()
    trainer.training_helpers().assign(model.unused, model.unused * float("nan"))
    # The active task does not read this parameter, so finite evaluation alone
    # cannot establish that the model is safe to persist.
    assert trainer.evaluate_downstream(tlx, model, batches(), trainer.resolve_dataset("aqsol"),
                                      None, "cpu")["metric"] == pytest.approx(1.5)
    with pytest.raises(FloatingPointError, match="checkpoint.*weight"):
        trainer.save_checkpoint(tlx, checkpoint, model, 2, 0.1)
    assert checkpoint.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [checkpoint]


@pytest.mark.parametrize("fault", ["nan", "cast_overflow"])
def test_tlx_checkpoint_rejects_nonfinite_payload_before_assigning(trainer, tmp_path, fault):
    model = ScalarModel()
    checkpoint = tmp_path / "best"
    trainer.save_checkpoint(tlx, checkpoint, model, 1, 0.25)
    if tlx.BACKEND == "tensorflow":
        with np.load(checkpoint, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        payload["0"] = np.full(payload["0"].shape,
                               float("nan") if fault == "nan" else 1e300, dtype=np.float64)
        with checkpoint.open("wb") as handle:
            np.savez(handle, **payload)
    else:
        import torch
        payload = torch.load(checkpoint, weights_only=False)
        payload["model"] = {name: torch.full(value.shape,
                             float("nan") if fault == "nan" else 1e300, dtype=torch.float64)
                            for name, value in payload["model"].items()}
        torch.save(payload, checkpoint)
    with pytest.raises(FloatingPointError, match="checkpoint.*weight"):
        trainer.restore_checkpoint(tlx, checkpoint, model, "cpu")
    np.testing.assert_array_equal(tlx.convert_to_numpy(model.weight), [1.0])


RDROP_REFERENCE = json.loads((Path(__file__).parent / "fixtures" /
                              "graph_tokenizer_rdrop_official.json").read_text())


class FixedRDropModel(ScalarModel):
    """Fixed stochastic realizations let both backends use the official oracle."""
    def __init__(self, case):
        super().__init__()
        self.case, self.calls = case, 0

    def forward(self, inputs, attention=None, task=None):
        branch = "first" if not self.case["enabled"] or self.calls % 2 == 0 else "second"
        self.calls += 1
        table = tlx.convert_to_tensor(self.case[branch], dtype=tlx.float32)
        return tlx.gather(table, inputs[:, 0]) * self.weight


def rdrop_batches(case, sizes):
    result, start = [], 0
    for size in sizes:
        labels = np.asarray(case["labels"][start:start + size]).reshape(size, -1)
        result.append((tlx.convert_to_tensor(np.arange(start, start + size).reshape(-1, 1), dtype=tlx.int64),
                       tlx.ones((size, 1), dtype=tlx.int64),
                       tlx.convert_to_tensor(labels, dtype=tlx.float32),
                       tlx.zeros((size,), dtype=tlx.int64)))
        start += size
    return result


@pytest.mark.parametrize("case", RDROP_REFERENCE["cases"],
                         ids=lambda c: f"{c['task']}-{c['enabled']}-{c['alpha']}")
@pytest.mark.parametrize("sizes,accumulation", [([3], 1), ([2, 1], 2), ([1, 1, 1], 2)])
def test_tlx_rdrop_matches_official_loss_and_gradient_with_tail_windows(trainer, case, sizes, accumulation):
    model = FixedRDropModel(case)
    optimizer = trainer.make_optimizer(0.0, 0.0)
    captured, steps = [], []
    original = optimizer.apply_gradients

    def apply(pairs):
        pairs = list(pairs)
        captured.append(float(tlx.convert_to_numpy(pairs[0][0]).reshape(-1)[0]))
        return original(pairs)

    optimizer.apply_gradients = apply
    config = args(gradient_accumulation_steps=accumulation,
                  finetune_use_consistency_regularization=case["enabled"],
                  finetune_consistency_alpha=case["alpha"])
    loss = trainer.train_downstream_epoch(
        tlx, model, rdrop_batches(case, sizes), optimizer,
        SimpleNamespace(step=lambda: steps.append(1)),
        trainer.resolve_dataset("aqsol" if case["task"] == "regression" else "dblp"),
        config, "cpu", seed=7, epoch=1)
    expected, start = [], 0
    for batch in range(0, len(sizes), accumulation):
        count = sum(sizes[batch:batch + accumulation])
        expected.append(np.mean(case["sample_weight_gradients"][start:start + count]))
        start += count
    assert loss == pytest.approx(np.mean(case["sample_losses"]), rel=2e-6)
    assert captured == pytest.approx(expected, rel=2e-6, abs=1e-7)
    assert len(steps) == len(expected)
    assert model.calls == len(sizes) * (2 if case["enabled"] else 1)


@pytest.mark.parametrize("preset", RDROP_REFERENCE["presets"],
                         ids=lambda p: f"{p['dataset']}-{p['encoder']}")
@pytest.mark.parametrize("override", [None, False, True])
def test_tlx_rdrop_official_presets_and_cli_reach_training(trainer, preset, override):
    argv = ["--dataset", preset["dataset"], "--encoder", preset["encoder"],
            "--finetune-noise-probability", "0", "--finetune-max-grad-norm", "1000000"]
    if override is not None:
        argv += ["--finetune-use-consistency-regularization" if override else
                 "--no-finetune-use-consistency-regularization"]
    config = trainer.apply_preset(trainer.build_parser().parse_args(argv))
    case = RDROP_REFERENCE["cases"][3 if (preset["enabled"] if override is None else override) else 0]
    if preset["dataset"] == "aqsol":
        # This test concerns whether the selected training branch is reached.
        case = {**case, "first": [[1.], [2.], [3.]], "second": [[2.], [3.], [4.]],
                "labels": [[0.], [0.], [0.]]}
    model = FixedRDropModel(case)
    optimizer = trainer.make_optimizer(0.0, 0.0)
    trainer.train_downstream_epoch(tlx, model, rdrop_batches(case, [3]), optimizer,
        SimpleNamespace(step=lambda: None), trainer.resolve_dataset(preset["dataset"]),
        config, "cpu", seed=7, epoch=1)
    assert model.calls == (2 if case["enabled"] else 1)
    assert config.finetune_consistency_alpha == preset["alpha"]


@pytest.mark.parametrize("alpha", ["nan", "inf", "-1"])
def test_rdrop_rejects_invalid_alpha_before_training(trainer, alpha):
    config = trainer.build_parser().parse_args(["--finetune-consistency-alpha", alpha])
    with pytest.raises(ValueError, match="consistency.*alpha"):
        trainer.apply_preset(config)


@pytest.mark.parametrize("fault", ["second_logits_nan", "second_logits_inf", "gradient", "total_loss"])
def test_tlx_rdrop_failure_keeps_main_incomplete_without_checkpoint(trainer, monkeypatch, tmp_path, fault):
    from gammagl.models import GraphBERT

    class FaultModel(GraphBERT):
        def __init__(self):
            super().__init__(64, 2, hidden_size=4, num_hidden_layers=1,
                             num_attention_heads=1, intermediate_size=8, max_position_embeddings=32)
            self.weight = self._get_weights('weight', shape=(1,), init=tlx.initializers.Constant(1.0))
            self.calls = 0

        def forward(self, inputs, attention=None, task=None):
            if task == "mlm":
                return tlx.ones((*inputs.shape, 64)) * self.weight
            self.calls += 1
            value = self.weight
            if self.is_train and self.calls % 2 == 0:
                if fault.startswith("second_logits"):
                    value = value * float(fault.rsplit("_", 1)[1])
                elif fault == "gradient":
                    value = tlx.sqrt(value - tlx.detach(value)) + 1.0
                value = -value
            values = tlx.ones((len(inputs), 1)) * value
            return tlx.concat((values, -values), axis=-1)

    monkeypatch.setattr(trainer, "make_model", lambda *_args: FaultModel())
    argv = ["--smoke", "--dataset", "mutagenicity", "--finetune-use-consistency-regularization",
            "--finetune-noise-probability", "0", "--seed", "42", "--output-dir", str(tmp_path)]
    if fault == "total_loss":
        argv += ["--finetune-consistency-alpha", "1e308"]
    with pytest.raises(FloatingPointError, match="finetune.*epoch=1.*batch=1"):
        trainer.main(argv)
    directory = tmp_path / "mutagenicity" / "bert"
    summary = json.loads((directory / "summary.json").read_text(), parse_constant=lambda s: pytest.fail(s))
    assert summary["complete"] is False and summary["run_metrics"] == []
    assert not list(directory.rglob("best_finetune.*"))
    assert not (directory / "run_01_seed_42" / "summary.json").exists()


def test_tokenizer_binds_official_vocabulary_row_mapping(trainer):
    reference = json.loads((Path(__file__).parent / "fixtures" / "graph_tokenizer_official.json").read_text())
    for case in reference["cases"]:
        config = trainer.apply_preset(trainer.build_parser().parse_args([
            "--dataset", case["dataset"], "--encoder", case["encoder"],
            "--bpe-merges", str(case["num_merges"]),
            "--num-serializations", str(case["num_serializations"])]))
        graphs = [trainer.GraphRecord(g["edge_index"], g["x"], g["edge_attr"], [0.]) for g in case["graphs"]]
        tokenizer = trainer.make_tokenizer(config, graphs)
        expected = [None] * (tokenizer.max_token_id + 1)
        for official, local in case["official_to_gammagl_model_id"].items():
            expected[local] = int(official)
        assert tokenizer.model_id_to_official_id == expected
        before = tokenizer.fingerprint()
        tokenizer.model_id_to_official_id[8:10] = tokenizer.model_id_to_official_id[8:10][::-1]
        assert tokenizer.fingerprint() != before


@pytest.mark.parametrize("swapped", [False, True])
def test_zero_only_corpus_keeps_complete_reserved_vocabulary_mapping(swapped):
    from gammagl.transforms import GraphTokenizer, GraphTokenizerSpecialTokens
    special = GraphTokenizerSpecialTokens(pad_token_id=1, unk_token_id=0) if swapped else GraphTokenizerSpecialTokens()
    tokenizer = GraphTokenizer(special_tokens=special).fit([
        dict(edge_index=[[], []], x=[0], num_nodes=1)])
    expected = [1, 0, 2, 3, 4, 5, 6, 7] if swapped else list(range(8))
    assert tokenizer.model_id_to_official_id == expected
    assert tokenizer.encode_graph(dict(edge_index=[[], []], x=[0], num_nodes=1)).input_ids == [3, special.pad_token_id, 4]


@pytest.mark.parametrize("dataset,mode", [("aqsol", "official_random"),
                                          ("mutagenicity", "official_pretrained"),
                                          ("dblp", "official_pretrained")])
def test_formal_gte_initialization_matches_each_preset(trainer, monkeypatch, tmp_path, dataset, mode):
    from gammagl.models import graph_gte_pretrained as pretrained
    fixture = json.loads((Path(__file__).parent / "fixtures" / "graph_gte_initialization_official.json").read_text())
    monkeypatch.setattr(pretrained, "load_official_gte_config", lambda **_k: fixture["config"])
    if mode == "official_random":
        monkeypatch.setattr(pretrained, "download_official_gte_checkpoint",
                            lambda **_k: pytest.fail("AQSOL official random path downloaded weights"))
    else:
        safetensors_numpy = pytest.importorskip("safetensors.numpy")
        checkpoint = tmp_path / "source.safetensors"
        safetensors_numpy.save_file({key: np.asarray(value, np.float32)
                                    for key, value in fixture["source"].items()}, str(checkpoint))
        monkeypatch.setattr(pretrained, "download_official_gte_checkpoint", lambda **_k: (checkpoint, pretrained.sha256_file(checkpoint)))
    config = trainer.apply_preset(trainer.build_parser().parse_args([
        "--dataset", dataset, "--encoder", "gte", "--max-length", "16",
        "--max-position-embeddings", "16", "--num-serializations", "1", "--bpe-merges", "0"]))
    graph = trainer.GraphRecord([[0, 1], [1, 0]], [1, 2], [0, 0], [0.])
    tokenizer = trainer.make_tokenizer(config, [graph])
    model = trainer.make_model(config, tokenizer.max_token_id + 1,
                               tokenizer.special_tokens.pad_token_id, 2, tokenizer)
    info = trainer.gte_initialization_summary(model, config)
    assert info["mode"] == mode
    assert info["pretrained"] is (mode == "official_pretrained")
    assert info["initialization_matches_preset"] is True
    assert info["reproduction"] is False
    assert model.model_id_to_official_id == tokenizer.model_id_to_official_id
    model.set_eval()
    ids = tlx.convert_to_tensor([tokenizer.encode_graph(graph).input_ids], dtype=tlx.int64)
    assert tuple(model(ids, tlx.ones_like(ids), task="supervised").shape) == (1, 2)
