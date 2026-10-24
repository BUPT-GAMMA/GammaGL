"""Fit-corpus provenance checks on both supported TLX backends."""
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import tensorlayerx as tlx

from test_graph_tokenizer_tlx_training import trainer, ScalarModel


def configured(trainer, tmp_path, dataset="aqsol", encoder="bert", scope=None):
    argv = ["--dataset", dataset, "--encoder", encoder, "--data-root", str(tmp_path / "data"),
            "--output-dir", str(tmp_path / "runs"), "--bpe-merges", "8",
            "--bpe-min-frequency", "2", "--bpe-backend", "python", "--num-serializations", "1"]
    if scope is not None:
        argv += ["--fit-scope", scope]
    return trainer.apply_preset(trainer.build_parser().parse_args(argv))


@pytest.mark.parametrize("dataset", ["aqsol", "mutagenicity", "dblp"])
@pytest.mark.parametrize("encoder", ["bert", "gte"])
def test_formal_presets_fit_all_and_allow_separate_train_variant(trainer, tmp_path, dataset, encoder):
    args = configured(trainer, tmp_path, dataset, encoder)
    assert args.fit_scope == "all"
    train_args = configured(trainer, tmp_path, dataset, encoder, "train")
    assert train_args.fit_scope == "train"
    spec = trainer.resolve_dataset(dataset)
    assert trainer.run_output_directory(args, spec, 0, 42) != trainer.run_output_directory(train_args, spec, 0, 42)


def test_fit_scope_binds_identical_corpus_and_checkpoint_before_weight_assignment(trainer, tmp_path):
    from gammagl.transforms import GraphTokenizer

    graph = dict(edge_index=[[0, 1], [1, 0]], x=[3, 7], edge_attr=[0, 0], num_nodes=2)
    train = GraphTokenizer().fit([graph], fit_scope="train")
    all_graphs = GraphTokenizer().fit([graph], fit_scope="all")
    assert train.vocabulary == all_graphs.vocabulary
    assert train.fingerprint() != all_graphs.fingerprint()
    assert GraphTokenizer().fit([graph]).fit_scope == "train"
    with pytest.raises(ValueError, match="fit_scope"):
        GraphTokenizer().fit([graph], fit_scope="invalid")
    source = ScalarModel()
    source.tokenizer_fingerprint = train.fingerprint()
    source.tokenizer_fit_scope = train.fit_scope
    target = ScalarModel()
    # Even a stale/forged fingerprint must not bypass explicit scope checking.
    target.tokenizer_fingerprint = train.fingerprint()
    target.tokenizer_fit_scope = all_graphs.fit_scope
    before = tlx.convert_to_numpy(target.weight).copy()
    checkpoint = tmp_path / ("best.npz" if tlx.BACKEND == "tensorflow" else "best.pt")
    trainer.save_checkpoint(tlx, checkpoint, source, 1, 0.5)
    with pytest.raises(ValueError, match="fit scope"):
        trainer.restore_checkpoint(tlx, checkpoint, target, "cpu")
    np.testing.assert_array_equal(tlx.convert_to_numpy(target.weight), before)


def test_cache_scope_revalidated_and_labels_remain_train_only(trainer, tmp_path, monkeypatch):
    args = configured(trainer, tmp_path)
    spec = trainer.resolve_dataset(args.dataset)
    splits = trainer.synthetic_splits(spec)
    # Include an ordinary token found only in held-out graphs.
    splits["test"][0].x = [91, 92, 93]
    monkeypatch.setattr(trainer, "load_dataset_splits", lambda *_: splits)
    all_tokenizer, _, normalizer, _ = trainer.load_or_prepare_preprocessing(args, spec)
    assert all_tokenizer.fit_scope == "all"
    assert {91, 92, 93} <= all_tokenizer.original_token_ids
    assert normalizer["mean"] == pytest.approx([0.1])
    all_path = trainer.preprocessing_cache_path(args, spec)
    train_args = configured(trainer, tmp_path, scope="train")
    train_tokenizer, _, train_normalizer, _ = trainer.load_or_prepare_preprocessing(train_args, spec)
    assert not {91, 92, 93} & train_tokenizer.original_token_ids
    assert train_normalizer == normalizer
    train_path = trainer.preprocessing_cache_path(train_args, spec)
    assert train_path != all_path
    # Forge the outer header while leaving a valid train-only tokenizer inside.
    payload = pickle.loads(train_path.read_bytes())
    payload["input_configuration"] = trainer.tokenizer_configuration(args)
    all_path.write_bytes(pickle.dumps(payload))
    rebuilt = trainer.load_or_prepare_preprocessing(args, spec)[0]
    assert rebuilt.fit_scope == "all"
    assert rebuilt.fingerprint() == all_tokenizer.fingerprint()
    all_path.unlink()
    splits["val"][0].y = [10000.]
    splits["test"][0].y = [-10000.]
    changed, _, changed_normalizer, _ = trainer.load_or_prepare_preprocessing(args, spec)
    assert changed.fingerprint() == all_tokenizer.fingerprint()
    assert changed_normalizer == normalizer


def test_aggregate_rejects_mixed_fit_scopes(trainer):
    def result(scope, seed):
        return dict(dataset="aqsol", encoder="bert", seed=seed, metric=0.5,
                    metric_name="MAE", tokenizer_fit_scope=scope)
    with pytest.raises(ValueError, match="fit scope"):
        trainer.aggregate_run_results([result("all", 42), result("train", 43)])
    assert trainer.aggregate_run_results([result("all", 42)])["tokenizer_fit_scope"] == "all"


def test_actual_official_prepare_pipeline_matches_native_fit_corpus(trainer, tmp_path):
    fixture = json.loads((Path(__file__).parent / "fixtures" / "graph_tokenizer_fit_scope_official.json").read_text())
    for case in fixture["cases"]:
        args = configured(trainer, tmp_path / f"{case['dataset']}-{case['encoder']}", case["dataset"], case["encoder"])
        spec = trainer.resolve_dataset(args.dataset)
        raw = Path(args.data_root) / spec.storage_name / "raw"
        raw.mkdir(parents=True)
        with (raw / "data.pkl").open("wb") as handle:
            pickle.dump([(dict(edge_index=g["edge_index"], node_token_ids=g["x"],
                               edge_token_ids=g["edge_attr"], num_nodes=len(g["x"])), float(i % 2))
                         for i, g in enumerate(case["graphs"])], handle)
        for split, indices in fixture["split_indices"].items():
            (raw / f"{split}_index.json").write_text(json.dumps(indices))
        tokenizer, encoded, _, _ = trainer.load_or_prepare_preprocessing(args, spec)
        assert tokenizer.fit_scope == "all"
        assert [list(rule) for rule in tokenizer.bpe.codebook.merge_rules] == case["merge_rules"]
        sequences = [r for split in ("train", "val", "test") for r in encoded[split]]
        assert [r["serialized_token_ids"] for r in sequences] == case["serialized"]
        mapping = tokenizer.model_id_to_official_id
        assert {str(token): mapping[model_id] for token, model_id in tokenizer.vocabulary.items()} == case["vocabulary"]
        assert [[mapping[i] for i in r["input_ids"]] for r in sequences] == case["input_ids"]
        assert [[list(key), value] for key, value in sorted(tokenizer.serializer.frequency_map.items())] == case["frequencies"]
        args.fit_scope = "train"
        train_tokenizer, train_encoded, _, _ = trainer.load_or_prepare_preprocessing(args, spec)
        assert train_tokenizer.bpe.codebook.merge_rules != tokenizer.bpe.codebook.merge_rules
        assert train_tokenizer.vocabulary != tokenizer.vocabulary
        if tokenizer.serializer.name == "feuler":
            assert train_tokenizer.serializer.frequency_map != tokenizer.serializer.frequency_map
            assert train_encoded["train"][1]["serialized_token_ids"] != encoded["train"][1]["serialized_token_ids"]
