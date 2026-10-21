"""Generate an offline oracle with the pinned, unmodified official BERT.

Usage: python generate_graph_bert_official.py /path/to/pinned/official
Requires transformers 4.40.2; no GammaGL imports or downloads.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import transformers
from transformers.models.bert import modeling_bert

root = Path(sys.argv[1]).resolve()
revision = '98343a6b025a48fbb6859cd812a12b81ec3ac3cc'
assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == revision
assert not subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD', '--', 'src'], text=True)
assert transformers.__version__ == '4.40.2'
sys.path.insert(0, str(root))
from src.models.unified_encoder import BertEncoder
from src.models.unified_task_head import UnifiedTaskHead

torch.set_num_threads(1)
torch.manual_seed(29)
config = dict(vocab_size=11, pad_token_id=0, hidden_size=4, num_hidden_layers=1,
              num_attention_heads=2, intermediate_size=6, hidden_dropout_prob=0.,
              attention_probs_dropout_prob=0., max_position_embeddings=8,
              layer_norm_eps=1e-12, type_vocab_size=2, initializer_range=.02,
              reset_weights=True)
encoder = BertEncoder('bert-small', **config)
mlm = UnifiedTaskHead(4, 'mlm', 11, {}, encoder.get_word_embeddings_weight())
task = UnifiedTaskHead(4, 'classification', 2,
                       dict(hidden_ratio=.5, activation='relu', dropout=0.))
encoder.eval()
mlm.eval()
task.eval()
ids = torch.tensor([[3, 0, 4, 0], [3, 8, 4, 9]])
attention = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
labels = torch.tensor([[-100, 8, 4, -100], [-100, 4, 8, -100]])
embedded = encoder.bert.embeddings(ids)
hidden = encoder.get_sequence_output(ids, attention)
mlm_logits = mlm(hidden)
task_logits = task(encoder.encode(ids, attention, 'mean'))
coefficients = torch.arange(hidden.numel()).reshape(hidden.shape) / 10.
loss = (torch.nn.functional.cross_entropy(mlm_logits.reshape(-1, 11), labels.reshape(-1))
        + (hidden * coefficients).sum()
        + torch.nn.functional.cross_entropy(task_logits, torch.tensor([0, 1])))
loss.backward()

def arrays(module, gradients=False):
    return {name: (weight.grad if gradients else weight).detach().tolist()
            for name, weight in module.named_parameters()
            if not gradients or weight.grad is not None}

sources = ['src/models/unified_encoder.py', 'src/models/unified_task_head.py',
           'src/models/bert/config.py']
result = dict(revision=revision, torch=torch.__version__, transformers=transformers.__version__,
              source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                             for name in sources},
              hf_modeling_bert_sha256=hashlib.sha256(Path(modeling_bert.__file__).read_bytes()).hexdigest(),
              config=config, encoder=arrays(encoder.bert), mlm=arrays(mlm.head), task=arrays(task.head),
              input_ids=ids.tolist(), attention_mask=attention.tolist(), labels=labels.tolist(),
              embedding_output=embedded.detach().tolist(), hidden_states=hidden.detach().tolist(),
              mlm_logits=mlm_logits.detach().tolist(), task_logits=task_logits.detach().tolist(),
              loss=loss.item(), encoder_gradients=arrays(encoder.bert, True),
              mlm_gradients=arrays(mlm.head, True), task_gradients=arrays(task.head, True))
Path(__file__).with_name('graph_bert_official.json').write_text(json.dumps(result, indent=2) + '\n')
