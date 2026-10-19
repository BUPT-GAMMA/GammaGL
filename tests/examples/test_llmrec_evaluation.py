"""LLMRec evaluation imports must not parse CLI arguments or read training data."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_evaluation_import_and_explicit_data_initialization(tmp_path):
    pytest.importorskip('sklearn', reason='LLMRec example evaluation requires scikit-learn')
    dataset = tmp_path / 'tiny'
    dataset.mkdir()
    for split, values in [('train', {'0': [0]}), ('val', {'0': [1]}), ('test', {'0': [2]})]:
        (dataset / f'{split}.json').write_text(json.dumps(values))
    np.save(dataset / 'text_feat.npy', np.ones((3, 2), dtype=np.float32))
    script = '''
import os, sys
from types import SimpleNamespace
sys.argv = ['embedding-client', '--not-an-llmrec-option']
from examples.llmrec.utility import batch_test
args = SimpleNamespace(data_path=os.environ['TEST_DATA_ROOT'],
                       dataset='tiny', batch_size=1, Ks='[1]', test_flag='part')
data = batch_test.configure(args)
assert data.n_users == 1 and data.n_items == 3
metrics = batch_test.test_one_user(([0., 0.9, 0.1], 0, True))
assert metrics['recall'].tolist() == [1.0]
assert metrics['precision'].tolist() == [1.0]
'''
    env = {**os.environ, 'TEST_DATA_ROOT': str(tmp_path) + '/',
           'PYTHONPATH': str(Path(__file__).resolve().parents[2])}
    result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
