"""Offline oracle: execute pinned GTEEncoder with tiny local model providers.

Usage: python generate_graph_gte_initialization_official.py /path/to/official
Only the Hugging Face config/checkpoint providers are replaced. The official
GTEEncoder, NewModel, resize, PAD handling and task head execute unmodified.
No GammaGL imports and no model/data downloads.
"""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
import transformers
from transformers.modeling_utils import PreTrainedModel

root = Path(sys.argv[1]).resolve()
revision = "98343a6b025a48fbb6859cd812a12b81ec3ac3cc"
assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == revision
assert not subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--", "src", "gte_model", "config"], text=True)
sys.path.insert(0, str(root))
from gte_model.configuration import NewConfig
from gte_model.modeling import NewModel
from src.models import unified_encoder
from src.models.unified_task_head import UnifiedTaskHead

config = dict(vocab_size=11, hidden_size=4, num_hidden_layers=1,
              num_attention_heads=2, intermediate_size=6, max_position_embeddings=16,
              hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
              layer_norm_eps=1e-12, type_vocab_size=1, pad_token_id=1,
              initializer_range=0.02, position_embedding_type="rope", pack_qkv=True,
              rope_theta=20000.0, rope_scaling={"factor": 8.0, "type": "ntk"},
              unpad_inputs=False, use_memory_efficient_attention=False)
torch.manual_seed(13)
original = NewModel(NewConfig(**config))
source = {"new." + name: value.tolist() for name, value in original.state_dict().items()}

cases = []
for reset, size in [(False, 9), (False, 13), (True, 9)]:
    torch.manual_seed(29)
    with patch.object(unified_encoder, "AutoConfig", SimpleNamespace(from_pretrained=lambda *_a, **_k: NewConfig(**config))), \
         patch.object(unified_encoder, "AutoModel", SimpleNamespace(
             from_pretrained=lambda *_a, **_k: copy.deepcopy(original),
             from_config=lambda cfg, **_k: NewModel(cfg))):
        encoder = unified_encoder.GTEEncoder("tiny-local", vocab_size=size, pad_token_id=0,
            reset_weights=reset, torch_dtype="fp32", unpad_inputs=False,
            use_memory_efficient_attention=False)
    model = encoder.gte_model
    model.eval()
    ids = torch.tensor([[3, 8, 4, 0], [3, 1, 4, 0]])
    attention = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
    hidden = model(ids, attention_mask=attention).last_hidden_state
    coefficients = torch.arange(hidden.numel(), dtype=torch.float32).reshape(hidden.shape) / 10.0
    loss = (hidden * coefficients).sum()
    loss.backward()
    head = UnifiedTaskHead(4, "mlm", size, {}, encoder.get_word_embeddings_weight())
    mapping = [1, 0, 2, 3, 4, 5, 6, 7] + list(reversed(range(8, size)))
    cases.append(dict(reset_weights=reset, vocab_size=size, model_id_to_official_id=mapping,
        input_ids=ids.tolist(), attention_mask=attention.tolist(),
        resized_embeddings=model.embeddings.word_embeddings.weight.detach().tolist(),
        embedding_output=model.embeddings(unpad_inputs=False, input_ids=ids)[0].detach().tolist(),
        hidden_states=hidden.detach().tolist(), loss=loss.item(),
        embedding_gradient=model.embeddings.word_embeddings.weight.grad.tolist(),
        qkv_gradient=model.encoder.layer[0].attention.qkv_proj.weight.grad.tolist(),
        mlm_weight=head.head.weight.detach().tolist(), mlm_bias=head.head.bias.detach().tolist(),
        mlm_logits=head(hidden.detach()).detach().tolist(),
        random_state={"new." + name: value.tolist() for name, value in model.state_dict().items()} if reset else None))

sources = ["src/models/unified_encoder.py", "src/models/unified_task_head.py",
           "gte_model/configuration.py", "gte_model/modeling.py"]
result = dict(revision=revision, torch=torch.__version__, transformers=transformers.__version__,
              resize_implementation_sha256=hashlib.sha256(inspect.getsource(PreTrainedModel._get_resized_embeddings).encode()).hexdigest(),
              config=config, source=source, cases=cases,
              source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources})
Path(__file__).with_name("graph_gte_initialization_official.json").write_text(json.dumps(result, indent=2) + "\n")
