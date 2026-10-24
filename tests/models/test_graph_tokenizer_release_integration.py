"""Opt-in real release/GTE integration; never used as benchmark evidence."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import tensorlayerx as tlx


def test_real_gte_checkpoint_training_and_offline_restore(tmp_path, record_property):
    data_root = os.environ.get('GAMMAGL_GRAPH_TOKENIZER_INTEGRATION_DATA_ROOT')
    cache = os.environ.get('GAMMAGL_GRAPH_TOKENIZER_GTE_CACHE')
    if not data_root or not cache:
        pytest.skip('set integration data root and pinned GTE cache to run real integration')
    if not Path(data_root).is_dir() or not Path(cache).is_dir():
        raise FileNotFoundError('Configured integration data root or GTE cache is missing')
    # Missing cache files must fail instead of starting a model download.
    from huggingface_hub import constants
    if not constants.HF_HUB_OFFLINE:
        pytest.fail('Start pytest with HF_HUB_OFFLINE=1 so a missing cache cannot download')
    from examples.graph_tokenizer import artifacts, graph_tokenizer_trainer as trainer
    from examples.graph_tokenizer.runtime import configure_device

    runtime = configure_device(tlx.BACKEND, 'cpu')
    args = trainer.apply_preset(trainer.build_parser().parse_args([
        '--dataset', 'mutagenicity', '--encoder', 'gte', '--device', 'cpu',
        '--data-root', data_root, '--gte-checkpoint-cache-dir', cache,
        '--fit-scope', 'train', '--bpe-merges', '8', '--num-serializations', '1',
        '--pretrain-epochs', '1', '--finetune-epochs', '1', '--batch-size', '1',
        '--mask-probability', '1',
        '--max-length', '128', '--no-finetune-use-consistency-regularization',
    ]))
    args.seed = 42
    args.validation_only = True
    trainer.set_seed(tlx, args.seed)
    spec = trainer.resolve_dataset(args.dataset)
    full = trainer.load_dataset_splits(data_root, spec)
    # Choose real, small graphs, keeping their release split membership.
    selected = {name: sorted(range(len(graphs)), key=lambda i: len(graphs[i].edge_attr))[:1]
                for name, graphs in full.items()}
    splits = {name: [full[name][i] for i in indices] for name, indices in selected.items()}
    args.integration_graph_indices = selected
    args.integration_split_sizes = {name: len(graphs) for name, graphs in splits.items()}
    tokenizer = trainer.fit_tokenizer(args, splits)
    encoded = trainer.encode_splits(tokenizer, splits, args)
    normalizer, width = trainer.normalize_labels(encoded, spec)
    model = trainer.make_model(args, tokenizer.max_token_id + 1,
                               tokenizer.special_tokens.pad_token_id, width, tokenizer)
    initialization = model.pretrained_manifest
    assert initialization['pretrained'] is True
    assert initialization['encoder_coverage'] == 1.0
    assert not initialization['missing'] and not initialization['unexpected']
    assert not initialization['shape_mismatches']
    from gammagl.models.graph_gte_pretrained import download_official_gte_checkpoint
    from safetensors import safe_open
    source, _ = download_official_gte_checkpoint(cache)
    model_id = next(i for i in range(model.vocab_size) if i != model.pad_token_id)
    official_id = tokenizer.model_id_to_official_id[model_id]
    with safe_open(str(source), framework='np', device='cpu') as handle:
        original_row = handle.get_slice('new.embeddings.word_embeddings.weight')[official_id:official_id + 1][0]
    np.testing.assert_array_equal(tlx.convert_to_numpy(model.token_embeddings.embeddings)[model_id], original_row)
    record_property('gte_initialization', json.dumps(initialization))
    model.tokenizer_fingerprint = tokenizer.fingerprint()
    model.tokenizer_fit_scope = tokenizer.fit_scope
    model.training_protocol_version = trainer.TRAINING_PROTOCOL_VERSION
    run = tmp_path / 'original' / 'run_01_seed_42'
    artifacts.save_run(run, args, spec, model, tokenizer, normalizer, encoded, runtime)
    loaders = {name: trainer.make_loader(tlx, rows, 1, tokenizer.special_tokens.pad_token_id,
                                         shuffle=False, choose_variant=False) for name, rows in encoded.items()}
    # Both stages exercise finite loss/gradient checks and actual optimizer updates.
    encoder_before = tlx.convert_to_numpy(model.encoder_layers[0].attention.qkv_proj.weights).copy()
    pretrain_checkpoint = run / ('best_pretrain.npz' if tlx.BACKEND == 'tensorflow' else 'best_pretrain.pt')
    mlm = trainer.run_mlm(tlx, model, loaders['train'], tokenizer, args, 'cpu',
                          val_loader=loaders['val'], checkpoint=pretrain_checkpoint)
    assert np.isfinite(mlm['final_loss']) and mlm['final_loss'] > 0
    assert not np.array_equal(encoder_before, tlx.convert_to_numpy(model.encoder_layers[0].attention.qkv_proj.weights))
    head_before = tlx.convert_to_numpy(model.task_head_out.weights).copy()
    checkpoint = run / ('best_finetune.npz' if tlx.BACKEND == 'tensorflow' else 'best_finetune.pt')
    finetune = trainer.run_finetuning(tlx, model, loaders['train'], loaders['val'],
                                     loaders['test'], spec, normalizer, args, 'cpu', checkpoint)
    assert not np.array_equal(head_before, tlx.convert_to_numpy(model.task_head_out.weights))
    summary = dict(seed=42, validation_only=True, mlm=mlm, finetune=finetune,
                   training_protocol=trainer.training_protocol(),
                   artifacts=artifacts.completed_run(run, checkpoint, pretrain_checkpoint=pretrain_checkpoint))
    artifacts.atomic_write(run / 'summary.json', artifacts.json_bytes(summary))
    ids = np.array([encoded['test'][0]['input_ids']], dtype=np.int64)
    expected = tlx.convert_to_numpy(model(tlx.convert_to_tensor(ids),
                                         tlx.ones(ids.shape, dtype=tlx.int64), task='supervised'))
    moved = tmp_path / 'moved'
    shutil.copytree(run.parent, moved)
    shutil.rmtree(run.parent)
    np.save(tmp_path / 'input.npy', ids)
    script = '''
import sys, numpy as np
from examples.graph_tokenizer.artifacts import load_run
restored = load_run(sys.argv[1], device='cpu')
import tensorlayerx as tlx
ids = tlx.convert_to_tensor(np.load(sys.argv[2]), dtype=tlx.int64)
result = restored['model'](ids, tlx.ones_like(ids), task='supervised')
np.save(sys.argv[3], tlx.convert_to_numpy(result))
'''
    env = {**os.environ, 'HF_HOME': str(tmp_path / 'empty-hf'), 'HF_HUB_OFFLINE': '1',
           'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(Path(__file__).resolve().parents[2])}
    output = tmp_path / 'actual.npy'
    restored = subprocess.run([sys.executable, '-c', script, str(moved / run.name),
                               str(tmp_path / 'input.npy'), str(output)],
                              env=env, capture_output=True, text=True, timeout=300)
    assert restored.returncode == 0, restored.stdout + restored.stderr
    np.testing.assert_allclose(np.load(output), expected, atol=1e-6, rtol=1e-6)
