"""Manual fixture generation; CI only reads the JSON, never imports this file.

Usage: TG_THREADS_FORCE=0 python generate_graph_tokenizer_official.py /path/to/official
Requires torch, DGL, PyYAML, pybind11, a C++ compiler and official runtime imports.
No GammaGL imports. Executes the unmodified official serializers, loader getters,
C++ minBPE, BPEEngine and VocabManager at the pinned commit.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import sysconfig
import tempfile
from types import SimpleNamespace

REVISION = "98343a6b025a48fbb6859cd812a12b81ec3ac3cc"
root = Path(sys.argv[1]).resolve()
assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == REVISION
assert not subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--", "src", "config"], text=True)
sys.path.insert(0, str(root))
os.environ["TG_THREADS_FORCE"] = "0"

import dgl
import networkx
import pybind11
import torch
import yaml

from src.algorithms.serializer.eulerian_serializer import EulerianSerializer
from src.algorithms.serializer.freq_eulerian_serializer import FeulerSerializer
from src.data.loader.aqsol_loader import AQSOLoader
from src.data.loader.dblp_loader import DBLPLoader
from src.data.loader.mutagenicity_loader import MutagenicityLoader
from src.algorithms.compression.bpe_engine import BPEEngine

# Import this leaf module directly to avoid the package's unrelated Transformer
# model imports; execute the complete, unmodified file.
vocab_spec = importlib.util.spec_from_file_location("official_vocab", root / "src/models/bert/vocab_manager.py")
vocab_module = importlib.util.module_from_spec(vocab_spec)
vocab_spec.loader.exec_module(vocab_module)
VocabManager = vocab_module.VocabManager
transform_spec = importlib.util.spec_from_file_location("official_transforms", root / "src/models/bert/transforms.py")
transform_module = importlib.util.module_from_spec(transform_spec)
transform_spec.loader.exec_module(transform_module)


def graph(nodes, edges):
    arcs = [(u, v, t) for u, v, t in edges] + [(v, u, t) for u, v, t in edges]
    return nodes, arcs


# Branching, tied neighbors, frequency guidance, cycles, connected/disconnected,
# a singleton and unequal component sizes. Include low ordinary token IDs.
specs = [
    graph([1, 2, 1, 3, 4], [(0, 1, 0), (0, 2, 1), (0, 4, 2), (1, 2, 0), (1, 3, 1)]),
    graph([1, 2, 3], [(0, 1, 1), (1, 2, 2), (2, 0, 1)]),
    graph([1, 1, 2, 4, 3, 2], [(0, 1, 0), (1, 2, 1), (3, 4, 2)]),
    graph([6], []),
    graph([1, 2, 1, 2], [(0, 1, 1), (1, 2, 1), (2, 3, 1)]),
    ([1, 2, 3], [(0, 1, 1), (1, 2, 1), (2, 0, 1)]),  # preserve one-way COO arcs
]

with tempfile.TemporaryDirectory() as temp:
    extension = Path(temp) / ("_cpp_bpe" + sysconfig.get_config_var("EXT_SUFFIX"))
    subprocess.run(["c++", "-O2", "-shared", "-std=c++17", "-fPIC",
        "-I" + pybind11.get_include(), "-I" + sysconfig.get_path("include"),
        str(root / "src/algorithms/compression/_cpp_bpe.cpp"), "-o", str(extension)], check=True)
    name = "src.algorithms.compression._cpp_bpe"
    spec = importlib.util.spec_from_file_location(name, extension)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    cases = []
    defaults = yaml.safe_load((root / "config/default_config.yml").read_text())
    sources = ["config/default_config.yml", "config.py", "src/utils/config_override.py",
        "prepare_data_new.py", "src/algorithms/serializer/base_serializer.py",
        "src/algorithms/serializer/eulerian_serializer.py",
        "src/algorithms/serializer/freq_eulerian_serializer.py",
        "src/algorithms/serializer/serializer_factory.py",
        "src/algorithms/compression/_cpp_bpe.cpp", "src/algorithms/compression/bpe_engine.py",
        "src/models/bert/vocab_manager.py", "src/models/bert/data.py", "src/models/bert/transforms.py",
        "src/training/evaluate.py", "src/training/task_handler.py"]
    for dataset, loader_cls in [("mutagenicity", MutagenicityLoader), ("dblp", DBLPLoader), ("aqsol", AQSOLoader)]:
        sources.append(f"src/data/loader/{dataset}_loader.py")
        for encoder in ["bert", "gte"]:
            config_source = "config/default_config.yml"
            method, mode = "feuler", "all"
            if encoder == "gte" and dataset != "aqsol":
                method = "eulerian" if dataset == "dblp" else "feuler"
                config_source = f"config/{dataset}/{method}/gte/config.json"
                overrides = json.loads((root / config_source).read_text())
                method = overrides["serialization"]["method"]
                mode = overrides["serialization"]["bpe"]["engine"]["encode_rank_mode"]
                sources.append(config_source)
            loader = loader_cls.__new__(loader_cls)
            loader.config = SimpleNamespace(component_sep_token_id=7)
            loader._cache_built = False
            loader.token_map = loader.get_token_map()
            data, graphs = [], []
            for nodes, arcs in specs:
                src, dst = [e[0] for e in arcs], [e[1] for e in arcs]
                edges = [e[2] for e in arcs]
                g = dgl.graph((src, dst), num_nodes=len(nodes))
                if dataset == "aqsol":
                    g.ndata["feat"] = torch.tensor(nodes)
                    g.edata["feat"] = torch.tensor(edges, dtype=torch.long)
                    x, edge_attr = [2 * n + 1 for n in nodes], [2 * e for e in edges]
                else:
                    g.ndata["node_token_ids"] = torch.tensor(nodes).view(-1, 1)
                    g.ndata["node_type_id"] = torch.tensor(nodes)
                    g.edata["edge_token_ids"] = torch.tensor(edges, dtype=torch.long).view(-1, 1)
                    g.edata["edge_type_id"] = torch.tensor(edges, dtype=torch.long)
                    x, edge_attr = nodes, edges
                data.append({"dgl_graph": g})
                graphs.append(dict(edge_index=[src, dst], x=x, edge_attr=edge_attr))
            serializer = (EulerianSerializer if method == "eulerian" else FeulerSerializer)()
            serializer.stats_parallel_enabled = False
            serializer.initialize_with_dataset(loader, data)
            outputs, sequences = [], []
            for sample in data:
                n = sample["dgl_graph"].num_nodes()
                count = min(3, n)
                starts = [i * max(1, n // count) for i in range(count)]
                result = serializer.multiple_serialize(sample, num_samples=3, parallel=False)
                raw = result.token_sequences
                assert raw == [serializer.serialize(sample, start_node=s).token_sequences[0] for s in starts]
                sequences.extend(raw)
                outputs.append(dict(starts=starts, serialized=raw))
            engine = BPEEngine(train_backend="cpp", encode_backend="cpp", encode_rank_mode="all")
            engine.train(sequences, num_merges=8, min_frequency=2)
            compressed = engine.batch_encode(sequences)
            # prepare_data_new always builds vocabulary with all-rank compressed data,
            # then injects every base/merged ID, even for raw downstream evaluation.
            all_ids = list({t for s in sequences for t in s} | {r[2] for r in engine.merge_rules})
            special = defaults["special_tokens"]
            params = {key + "_token": special[key] for key in ["pad", "unk", "mask", "cls", "sep", "node_start", "node_end", "component_sep"]}
            params.update({key + "_token_id": val for key, val in special["ids"].items()})
            # Node start/end names use *_token_id, as do all other entries.
            vocab = VocabManager(**params)
            vocab.add_token_sequences(compressed + [all_ids])
            vocab.build_vocab(min_freq=1, max_vocab_size=None)
            gamma_vocab = {token: 8 + i for i, token in enumerate(sorted(set(all_ids) - {0}))}
            if 0 in all_ids:
                gamma_vocab[0] = 1 if encoder == "gte" else 0
            mapping = {i: i for i in range(8)}
            if encoder == "gte":
                mapping[0], mapping[1] = 1, 0
            mapping.update({vocab.token_to_id[token]: gamma_vocab[token] for token in all_ids})
            assert len(set(mapping.values())) == len(mapping)
            assert set(mapping) == set(range(vocab.vocab_size))
            for output in outputs:
                output["bpe_encoded"] = engine.batch_encode(output["serialized"])
                model_tokens = output["serialized"] if mode == "none" else output["bpe_encoded"]
                output["input_ids"] = [vocab.encode_sequence(s)["input_ids"].tolist() for s in model_tokens]
                output["augmentation"] = []
                for seed in [0, 3, 7]:
                    random.seed(seed)
                    swapped = transform_module.RandomSwap(swap_ratio=0.5, probability=1., window_size=3)(output["serialized"][0])
                    masked = transform_module.SequenceMasking(mask_ratio=0.3, probability=1., mask_token_id=2)(swapped)
                    encoded = masked if mode == "none" else engine.encode(masked)
                    output["augmentation"].append(dict(seed=seed, swapped=swapped, serialized=masked,
                        input_ids=vocab.encode_sequence(encoded)["input_ids"].tolist()))
            cases.append(dict(dataset=dataset, encoder=encoder, config_source=config_source,
                serializer=method, bpe_encode_mode=mode, num_merges=8, min_frequency=2,
                num_serializations=3, graphs=graphs, outputs=outputs,
                merge_rules=engine.merge_rules, official_to_gammagl_model_id=mapping))
    result = dict(repository="BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers",
        revision=REVISION, python=platform.python_version(), torch=torch.__version__, dgl=dgl.__version__,
        networkx=networkx.__version__,
        augmentation_config=dict(swap_probability=1., swap_ratio=0.5, swap_window=3,
                                 mask_probability=1., mask_ratio=0.3, mask_token_id=2),
        source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sorted(set(sources))},
        cases=cases)
    rendered = json.dumps(result, indent=2)
    # Keep scalar token arrays on one line for reviewable, small fixtures.
    rendered = re.sub(r"\[\s*(-?\d+(?:\s*,\s*-?\d+)*)\s*\]",
                      lambda match: "[" + re.sub(r"\s+", "", match[1]) + "]", rendered)
    Path(__file__).with_name("graph_tokenizer_official.json").write_text(rendered + "\n")
