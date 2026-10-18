"""Offline comparisons against executed, pinned official source (see fixtures)."""
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_graph_tokenizer_pipeline import trainer, _record, _preprocessing_args, _write_preprocessing_raw


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6])
def test_classification_avg_softmax_before_grouping(trainer, batch_size):
    import torch

    # Three variants cross batches; graph IDs are interleaved and have unequal counts.
    logits = [[10., 0.], [0., 5.], [0., 2.], [5., 0.], [0., 2.], [0., 5.]]
    gids = [8, 3, 8, 12, 8, 3]
    labels = [1, 1, 1, 0, 1, 1]
    assert np.asarray([logits[i] for i in [0, 2, 4]]).mean(0).argmax() == 0

    class Model(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None, task=None):
            return torch.tensor([logits[int(row[1]) - 8] for row in input_ids])

    records = [dict(_record(gid, [float(label)]), input_ids=[3, 8 + i, 4])
               for i, (gid, label) in enumerate(zip(gids, labels))]
    for phase in ["validation", "test"]:
        loader = trainer.make_loader(torch, records, batch_size, 0, False, False)
        result = trainer.evaluate_downstream(torch, Model(), loader,
                    trainer.resolve_dataset("dblp"), None, "cpu", phase)
        assert result["num_graphs"] == 3
        assert result["accuracy"] == 1.0


@pytest.mark.parametrize("dataset,encoder,method,mode", [
    ("mutagenicity", "bert", "feuler", "all"),
    ("mutagenicity", "gte", "feuler", "none"),
    ("dblp", "bert", "feuler", "all"),
    ("dblp", "gte", "eulerian", "none"),
    ("aqsol", "bert", "feuler", "all"),
    ("aqsol", "gte", "feuler", "all"),
])
def test_preset_reaches_real_serializer(trainer, dataset, encoder, method, mode):
    args = trainer.apply_preset(trainer.build_parser().parse_args([
        "--dataset", dataset, "--encoder", encoder, "--num-serializations", "1",
        "--bpe-merges", "2"]), trainer.PAPER_CONFIGS[(dataset, encoder)])
    tokenizer = trainer.make_tokenizer(args, [trainer.GraphRecord(
        [[0, 1], [1, 0]], [13, 17], [0, 0], [0.])])
    assert tokenizer.serializer.name == method
    assert tokenizer.serializer.protocol == "official"
    assert tokenizer.bpe_encode_mode == mode
    assert tokenizer.serializer.serialize(
        trainer.GraphRecord([[0, 1], [1, 0]], [13, 17], [0, 0], [0.])).token_ids == (
            [13, 0, 17, 0, 13] if dataset == "aqsol" else [27, 35, 27])


def test_official_stream_cannot_restore_node_identity():
    from gammagl.transforms.graph_tokenizer import GraphTokenizer
    graph = dict(edge_index=[[0, 1], [1, 0]], x=[9, 9], edge_attr=[0, 0], num_nodes=2)
    tokenizer = GraphTokenizer().fit([graph])
    with pytest.raises(NotImplementedError, match="node identit|reconstruct"):
        tokenizer.decode_graph(tokenizer.encode_graph(graph))


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_official_fixtures(trainer, backend):
    if backend == "cpp":
        from third_party import graph_bpe_cpp
        if not graph_bpe_cpp.is_available():
            pytest.skip("optional native BPE extension is not built")
    fixture = json.loads((Path(__file__).parent / "fixtures" / "graph_tokenizer_official.json").read_text())
    for case in fixture["cases"]:
        args = trainer.apply_preset(trainer.build_parser().parse_args([
            "--dataset", case["dataset"], "--encoder", case["encoder"],
            "--bpe-merges", str(case["num_merges"]),
            "--bpe-backend", backend,
            "--num-serializations", str(case["num_serializations"])]),
            trainer.PAPER_CONFIGS[(case["dataset"], case["encoder"])])
        graphs = [trainer.GraphRecord(g["edge_index"], g["x"], g["edge_attr"], [0.])
                  for g in case["graphs"]]
        tokenizer = trainer.make_tokenizer(args, graphs)
        assert [list(rule) for rule in tokenizer.bpe.codebook.merge_rules] == case["merge_rules"]
        # Fixed bijection for model IDs, including PAD/UNK swap for GTE.
        mapping = {int(k): v for k, v in case["official_to_gammagl_model_id"].items()}
        for graph, expected in zip(graphs, case["outputs"]):
            assert tokenizer.realization_start_nodes(graph, args.num_serializations) == expected["starts"]
            for start, raw, encoded, ids in zip(expected["starts"], expected["serialized"],
                    expected["bpe_encoded"], expected["input_ids"]):
                result = tokenizer.encode_graph(graph, start_node=start)
                assert tokenizer.serializer.serialize(graph, start_node=start).token_ids == raw
                assert result.serialized_token_ids == raw
                assert tokenizer.bpe.encode(raw) == encoded
                assert result.input_ids == [mapping[i] for i in ids]
            augmentation = fixture["augmentation_config"]
            for augmented in expected["augmentation"]:
                assert trainer.random_swap_serialized_tokens(expected["serialized"][0], set(),
                    1., 0.5, 3, random.Random(augmented["seed"])) == augmented["swapped"]
                record = dict(input_ids=[mapping[i] for i in expected["input_ids"][0]],
                              serialized_token_ids=expected["serialized"][0], graph_id=0, labels=[0.])
                loader = trainer.make_loader(None, [record], 1, tokenizer.special_tokens.pad_token_id,
                    False, False, tokenizer, augmentation, 256, augmented["seed"])
                batch = next(iter(loader))
                assert batch[0].tolist()[0] == [mapping[i] for i in augmented["input_ids"]]
        batched = tokenizer.batch_encode_graphs(graphs)
        assert [r.input_ids for r in batched] == [tokenizer.encode_graph(g).input_ids for g in graphs]
        assert len(set(mapping.values())) == len(mapping)
        assert set(mapping) == set(mapping.values()) == set(range(tokenizer.max_token_id + 1))


@pytest.mark.parametrize("change,value", [
    ("serializer", "eulerian"), ("token_protocol", "local_refs"), ("bpe_encode_mode", "none")])
def test_cache_and_checkpoint_bind_actual_input_protocol(trainer, tmp_path, change, value):
    import pickle
    import torch

    root, cache = tmp_path / "data", tmp_path / "cache"
    _write_preprocessing_raw(root)
    args = _preprocessing_args(root, cache)
    # Both representations are available to official Eulerian edge lookups.
    raw = root / "aqsol/raw/data.pkl"
    with raw.open("rb") as handle:
        samples = pickle.load(handle)
    for graph, _ in samples:
        graph["edge_index"], graph["edge_attr"] = [[0, 1], [1, 0]], [1, 1]
    with raw.open("wb") as handle:
        pickle.dump(samples, handle)
    spec = trainer.resolve_dataset("aqsol")
    original = trainer.load_or_prepare_preprocessing(args, spec)[0]
    original_path = trainer.preprocessing_cache_path(args, spec)
    setattr(args, change, value)
    altered = trainer.load_or_prepare_preprocessing(args, spec)[0]
    altered_path = trainer.preprocessing_cache_path(args, spec)
    assert altered_path != original_path
    assert original.fingerprint() != altered.fingerprint()
    # Even moving an otherwise current cache payload to a different key is refused.
    original_bytes = original_path.read_bytes()
    altered_path.write_bytes(original_bytes)
    rebuilt = trainer.load_or_prepare_preprocessing(args, spec)[0]
    assert rebuilt.fingerprint() == altered.fingerprint()
    assert altered_path.read_bytes() != original_bytes
    # A current payload with a modified vocabulary must also fail fingerprint validation.
    payload = pickle.loads(altered_path.read_bytes())
    payload["tokenizer"].vocabulary[99999] = 99999
    altered_path.write_bytes(pickle.dumps(payload))
    assert trainer.load_or_prepare_preprocessing(args, spec)[0].fingerprint() == altered.fingerprint()

    model = torch.nn.Linear(2, 2)
    model.tokenizer_fingerprint = original.fingerprint()
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(torch, path, model, 1, 1.)
    trainer.restore_checkpoint(torch, path, model, "cpu")
    before = model.weight.detach().clone()
    model.tokenizer_fingerprint = altered.fingerprint()
    with pytest.raises(ValueError, match="fingerprint"):
        trainer.restore_checkpoint(torch, path, model, "cpu")
    assert torch.equal(model.weight, before)


@pytest.mark.parametrize("field,value", [("protocol", "local_refs"), ("name", "eulerian")])
def test_fitted_tokenizer_rejects_serializer_mutation(field, value):
    from gammagl.transforms.graph_tokenizer import GraphTokenizer
    graph = dict(edge_index=[[0, 1], [1, 0]], x=[9, 9], edge_attr=[0, 0], num_nodes=2)
    tokenizer = GraphTokenizer().fit([graph])
    setattr(tokenizer.serializer, field, value)
    with pytest.raises(RuntimeError, match="configuration"):
        tokenizer.encode_graph(graph)


def test_regression_avg_with_unequal_variants_and_batch_boundaries(trainer):
    import torch

    class Model(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None, task=None):
            return (input_ids[:, 1:2].float() - 8.)

    # Graph 8 averages normalized predictions (0,2) to 1; graph 3 has one 4.
    records = [dict(_record(8, [0.]), input_ids=[3, 8, 4]),
               dict(_record(3, [2.]), input_ids=[3, 12, 4]),
               dict(_record(8, [0.]), input_ids=[3, 10, 4])]
    for size in [1, 2, 3]:
        result = trainer.evaluate_downstream(torch, Model(),
            trainer.make_loader(torch, records, size, 0, False, False),
            trainer.resolve_dataset("aqsol"), dict(mean=[10.], std=[2.]), "cpu")
        assert result["num_graphs"] == 2
        assert result["mae"] == 3.  # Per-graph raw errors 2 and 4.


def test_official_eulerian_does_not_replace_feuler_by_name_only():
    from gammagl.transforms.graph_serializer import EulerianSerializer, FrequencyGuidedEulerianSerializer
    graph = dict(edge_index=[[0, 1, 0, 2], [1, 0, 2, 0]], x=[11, 13, 17],
                 edge_attr=[0, 0, 0, 0], num_nodes=3)
    assert EulerianSerializer().fit([graph]).serialize(graph).token_ids == [11, 17, 11, 13, 11]
    assert FrequencyGuidedEulerianSerializer().fit([graph]).serialize(graph).token_ids == [11, 13, 11, 17, 11]


@pytest.mark.parametrize("mode", ["all", "none"])
def test_unseen_raw_labels_cannot_impersonate_learned_bpe_ids(mode):
    from gammagl.transforms.graph_tokenizer import GraphTokenizer
    from gammagl.transforms.graph_serializer import FrequencyGuidedEulerianSerializer
    from gammagl.transforms.graph_bpe import GraphBPE

    train = dict(edge_index=[[0, 1], [1, 0]], x=[3, 3], edge_attr=[2, 2], num_nodes=2)
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(dataset="aqsol"),
        bpe=GraphBPE(num_merges=1), bpe_encode_mode=mode).fit([train])
    assert tokenizer.bpe.codebook.merge_rules == [(3, 3, 5)]
    held_out = dict(edge_index=[[], []], x=[5], edge_attr=[], num_nodes=1)
    result = tokenizer.encode_graph(held_out)
    assert result.serialized_token_ids == [5]
    assert result.input_ids == [3, tokenizer.special_tokens.unk_token_id, 4]
    assert tokenizer.batch_encode_graphs([held_out])[0].input_ids == result.input_ids


@pytest.mark.parametrize("mode", ["all", "none"])
@pytest.mark.parametrize("gte", [False, True])
def test_official_zero_always_aliases_pad_even_when_absent_from_training(mode, gte):
    from gammagl.transforms.graph_tokenizer import GraphTokenizer, GraphTokenizerSpecialTokens
    from gammagl.transforms.graph_serializer import FrequencyGuidedEulerianSerializer

    train = dict(edge_index=[[0, 1], [1, 0]], x=[0, 1], edge_attr=[0, 0], num_nodes=2)
    special = GraphTokenizerSpecialTokens(pad_token_id=1, unk_token_id=0) if gte else GraphTokenizerSpecialTokens()
    tokenizer = GraphTokenizer(serializer=FrequencyGuidedEulerianSerializer(dataset="dblp"),
        special_tokens=special, bpe_encode_mode=mode).fit([train])
    assert 0 not in tokenizer.original_token_ids
    held_out = dict(edge_index=[[], []], x=[0], edge_attr=[], num_nodes=1)
    assert tokenizer.encode_graph(held_out).input_ids == [3, special.pad_token_id, 4]
