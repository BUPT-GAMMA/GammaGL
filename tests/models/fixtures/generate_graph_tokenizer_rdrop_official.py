"""Generate offline R-Drop loss/gradient references from unmodified official code.

Usage: python generate_graph_tokenizer_rdrop_official.py /path/to/pinned/official
Requires torch and PyYAML; does not import GammaGL or download models/data.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml


REVISION = "98343a6b025a48fbb6859cd812a12b81ec3ac3cc"
root = Path(sys.argv[1]).resolve()
assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == REVISION
assert not subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--", "src", "config"], text=True)
sys.path.insert(0, str(root))
from src.training.task_handler import TaskHandler


sources = ["src/training/task_handler.py", "config/default_config.yml",
           "config/mutagenicity/feuler/gte/config.json", "config/dblp/eulerian/gte/config.json"]
presets = []
default = yaml.safe_load((root / sources[1]).read_text())["bert"]["finetuning"]["augmentation_config"]
for dataset in ["mutagenicity", "dblp", "aqsol"]:
    for encoder in ["bert", "gte"]:
        config = default
        if encoder == "gte" and dataset != "aqsol":
            method = "eulerian" if dataset == "dblp" else "feuler"
            config = json.loads((root / f"config/{dataset}/{method}/gte/config.json").read_text())["bert"]["finetuning"]["augmentation_config"]
        presets.append(dict(dataset=dataset, encoder=encoder,
                            enabled=config["use_consistency_regularization"],
                            alpha=config["consistency_alpha"]))

cases = []
for task, enabled, alpha in [("classification", False, 1.0), ("classification", True, 0.0),
                             ("classification", True, 0.7), ("classification", True, 1.0),
                             ("regression", True, 0.7)]:
    first = [[2.0, -1.0], [0.5, 1.5], [-2.0, 1.0]]
    second = [[0.25, 1.75], [2.0, -0.5], [1.0, -2.0]]
    labels = [0, 1, 0]
    if task == "regression":
        first, second, labels = [[2.0], [0.5], [-2.0]], [[0.25], [2.0], [1.0]], [[0.0], [1.0], [0.0]]
    handler = TaskHandler(task, len(first[0]))
    sample_losses, sample_weight_gradients = [], []
    for index in range(len(labels)):
        weight = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        a = torch.tensor(first[index:index + 1], dtype=torch.float64) * weight
        b = torch.tensor(second[index:index + 1], dtype=torch.float64) * weight
        y = torch.tensor(labels[index:index + 1])
        loss = (handler.compute_loss_with_consistency(a, b, y, alpha)[0] if enabled
                else handler.compute_loss(a, y))
        sample_losses.append(loss.item())
        sample_weight_gradients.append(torch.autograd.grad(loss, weight)[0].item())
    cases.append(dict(task=task, enabled=enabled, alpha=alpha, first=first, second=second,
                      labels=labels, sample_losses=sample_losses,
                      sample_weight_gradients=sample_weight_gradients))

result = dict(revision=REVISION, torch=torch.__version__,
              source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
              presets=presets, cases=cases)
Path(__file__).with_name("graph_tokenizer_rdrop_official.json").write_text(
    json.dumps(result, indent=2, allow_nan=False) + "\n")
