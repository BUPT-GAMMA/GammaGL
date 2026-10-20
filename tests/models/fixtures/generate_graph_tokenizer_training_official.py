"""Offline sampling/masking oracle from the unmodified pinned official data module.

Usage: python generate_graph_tokenizer_training_official.py OFFICIAL_ROOT
Requires torch, NumPy, scikit-learn and PyYAML; no model/data downloads.
Random draws and the no-BPE provider are controlled; official algorithms are not edited.
"""
import hashlib
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys
import types
from unittest.mock import patch

import torch
import yaml

REVISION = '98343a6b025a48fbb6859cd812a12b81ec3ac3cc'
root = Path(sys.argv[1]).resolve()
assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
assert not subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD', '--', 'src', 'config'], text=True)
# Load the complete official leaf package without unrelated model dependencies.
package = types.ModuleType('official_training_data')
package.__path__ = [str(root / 'src/models/bert')]
sys.modules[package.__name__] = package
data = importlib.import_module(package.__name__ + '.data')
vocab_class = importlib.import_module(package.__name__ + '.vocab_manager').VocabManager
defaults = yaml.safe_load((root / 'config/default_config.yml').read_text())
special = defaults['special_tokens']
kwargs = {name + '_token': special[name] for name in special['ids']}
kwargs.update({name + '_token_id': value for name, value in special['ids'].items()})
vocab = vocab_class(**kwargs)
sequences = [[8], [9], [10], [11], [12], [13]]
graph_ids = [4, 9, 4, 2, 9, 4]
vocab.add_token_sequences(sequences)
vocab.build_vocab(min_freq=1, max_vocab_size=None)
dataset = data.ClassificationDataset(sequences, [1] * 6, vocab, data.NoOpTransform(),
                                     3, graph_ids, group_by_graph=True, variant_selection='random')
dataset._bpe_checked = True
dataset._bpe_transform = types.SimpleNamespace(encode=lambda tokens: tokens)
sampling = []
for seed in (1, 7, 42):
    random.seed(seed)
    sampling.append(dict(seed=seed, epochs=[
        [dataset[i]['input_ids'].tolist() for i in range(len(dataset))] for _ in range(3)]))

# Exercise every special-ID case and all three corruption branches, with an
# attention-masked ordinary token. Bernoulli providers obey the actual probabilities.
ids = torch.tensor([[3, 8, 9, 10, 7, 2, 1, 5, 6, 4, 0, 11]])
attention = torch.tensor([[1] * 11 + [0]])
draws = [torch.zeros(ids.shape),
         torch.tensor([[0., .1, .9, .9, .9, .1, .1, .1, .1, 0., 0., 0.]]),
         torch.tensor([[0., 0., .1, .9, .1, 0., 0., 0., 0., 0., 0., 0.]])]
replacements = torch.tensor([[0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
mlm = data.MLMDataset(sequences, vocab, data.NoOpTransform(), mlm_probability=1.)
iterator = iter(draws)
with patch.object(torch, 'bernoulli', side_effect=lambda probabilities: (next(iterator) < probabilities).float()), \
        patch.object(torch, 'randint', return_value=replacements):
    masked, labels = mlm._create_mlm_mask(ids.clone(), attention)

sources = ['src/models/bert/data.py', 'src/models/bert/vocab_manager.py',
           'src/training/tasks.py', 'src/training/loops.py', 'src/training/pretrain_pipeline.py',
           'config/default_config.yml', 'config/mutagenicity/feuler/gte/config.json',
           'config/dblp/eulerian/gte/config.json']
presets = []
for dataset_name in ('mutagenicity', 'dblp', 'aqsol'):
    for encoder in ('bert', 'gte'):
        settings = defaults
        if encoder == 'gte' and dataset_name != 'aqsol':
            method = 'eulerian' if dataset_name == 'dblp' else 'feuler'
            settings = json.loads((root / f'config/{dataset_name}/{method}/gte/config.json').read_text())
        presets.append(dict(dataset=dataset_name, encoder=encoder,
            pretrain=settings['bert']['pretraining']['apply_graph_level_to_val'],
            finetune=settings['bert']['finetuning']['apply_graph_level_to_val'],
            variant_selection=settings['bert']['finetuning']['graph_variant_selection'],
            pretrain_patience=settings['bert']['pretraining']['early_stopping_patience']))
result = dict(revision=REVISION, torch=torch.__version__, presets=presets,
    source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
    records=[dict(input_ids=vocab.encode_sequence(s)['input_ids'].tolist(), graph_id=gid, labels=[1.])
             for s, gid in zip(sequences, graph_ids)], sampling=sampling,
    masking=dict(input_ids=ids.tolist(), attention=attention.tolist(), draws=[d.tolist() for d in draws],
                 replacements=replacements.tolist(), vocab_size=vocab.vocab_size,
                 masked=masked.tolist(), labels=labels.tolist()))
Path(__file__).with_name('graph_tokenizer_training_official.json').write_text(
    json.dumps(result, indent=2, allow_nan=False) + '\n')
