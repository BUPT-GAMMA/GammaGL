"""Generate the offline oracle by running the pinned official preparation entry.

Usage: TG_THREADS_FORCE=0 python generate_graph_tokenizer_fit_scope_official.py OFFICIAL_ROOT
Requires the paper environment, pybind11 and a C++ compiler. No GammaGL imports.
Only ProjectConfig construction is redirected to a temporary CPU configuration;
loaders, corpus selection, serialization, C++ BPE and vocabulary building execute
the unmodified official source. The serializer factory spy only retains objects.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import sysconfig
import tempfile
from unittest.mock import patch

REVISION = "98343a6b025a48fbb6859cd812a12b81ec3ac3cc"
root = Path(sys.argv[1]).resolve()
assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == REVISION
assert not subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--", "src", "config", "config.py", "prepare_data_new.py"], text=True)
sys.path.insert(0, str(root))
os.environ["TG_THREADS_FORCE"] = "0"

import dgl
import pybind11
import torch
import yaml

import config as config_module
from config import ProjectConfig
from prepare_data_new import _prepare_one_mp
from src.algorithms.serializer.serializer_factory import SerializerFactory
from src.data.unified_data_interface import UnifiedDataInterface

split_indices = dict(train=[2, 0], val=[3], test=[1])
fit_order = [2, 0, 3, 1]
# Held-out graph 3 reverses the frequency preference at graph 0's branch;
# graph 1 contributes otherwise unseen tokens. Splits interleave raw indices.
specs = [([1, 2, 3], [(0, 1), (0, 2)]),
         ([11, 12], [(0, 1)]),
         ([4, 4, 4], [(0, 1), (1, 2)]),
         ([1, 3, 3], [(0, 1), (0, 2)])]
sources = ["config.py", "config/default_config.yml", "prepare_data_new.py",
           "src/data/base_loader.py", "src/data/unified_data_interface.py",
           "src/data/unified_data_factory.py", "src/algorithms/serializer/serializer_factory.py",
           "src/algorithms/serializer/base_serializer.py",
           "src/algorithms/serializer/eulerian_serializer.py",
           "src/algorithms/serializer/freq_eulerian_serializer.py",
           "src/algorithms/compression/bpe_engine.py", "src/algorithms/compression/_cpp_bpe.cpp",
           "src/models/bert/vocab_manager.py"]

with tempfile.TemporaryDirectory() as temp:
    temporary = Path(temp)
    extension = temporary / ("_cpp_bpe" + sysconfig.get_config_var("EXT_SUFFIX"))
    subprocess.run(["c++", "-O2", "-shared", "-std=c++17", "-fPIC",
                    "-I" + pybind11.get_include(), "-I" + sysconfig.get_path("include"),
                    str(root / "src/algorithms/compression/_cpp_bpe.cpp"), "-o", str(extension)], check=True)
    name = "src.algorithms.compression._cpp_bpe"
    module_spec = importlib.util.spec_from_file_location(name, extension)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    cases = []
    for dataset in ("aqsol", "mutagenicity", "dblp"):
        sources.append(f"src/data/loader/{dataset}_loader.py")
        for encoder in ("bert", "gte"):
            directory = temporary / f"{dataset}-{encoder}"
            directory.mkdir()
            settings = yaml.safe_load((root / "config/default_config.yml").read_text())
            settings["system"]["device"] = "cpu"
            settings["paths"]["project_root"] = str(directory)
            settings["dataset"]["name"] = dataset
            config_file = directory / "config.yml"
            config_file.write_text(yaml.safe_dump(settings))
            config = ProjectConfig(str(config_file))
            method = "eulerian" if (dataset, encoder) == ("dblp", "gte") else "feuler"
            mode = "none" if encoder == "gte" and dataset != "aqsol" else "all"
            if encoder == "gte" and dataset != "aqsol":
                source = f"config/{dataset}/{method}/gte/config.json"
                sources.append(source)
                overrides = json.loads((root / source).read_text())
                assert overrides["serialization"]["method"] == method
                assert overrides["serialization"]["bpe"]["engine"]["encode_rank_mode"] == mode
            data_directory = Path(config.data_dir) / dataset
            data_directory.mkdir(parents=True)
            raw, graphs = [], []
            for index, (nodes, edges) in enumerate(specs):
                arcs = edges + [(v, u) for u, v in edges]
                src, dst = [u for u, v in arcs], [v for u, v in arcs]
                graph = dgl.graph((src, dst), num_nodes=len(nodes))
                edge_types = [1 if dataset == "aqsol" else 0] * len(arcs)
                if dataset == "aqsol":
                    graph.ndata["feat"] = torch.tensor(nodes)
                    graph.edata["feat"] = torch.tensor(edge_types)
                    x, attr = [2 * n + 1 for n in nodes], [2 * e for e in edge_types]
                else:
                    graph.ndata["node_token_ids"] = torch.tensor(nodes).view(-1, 1)
                    graph.ndata["node_type_id"] = torch.tensor(nodes)
                    graph.edata["edge_token_ids"] = torch.tensor(edge_types).view(-1, 1)
                    graph.edata["edge_type_id"] = torch.tensor(edge_types)
                    x, attr = nodes, edge_types
                raw.append((graph, float(index % 2) if dataset == "aqsol" else index % 2))
                graphs.append(dict(edge_index=[src, dst], x=x, edge_attr=attr))
            with (data_directory / "data.pkl").open("wb") as handle:
                pickle.dump(raw, handle)
            for split, indices in split_indices.items():
                (data_directory / f"{split}_index.json").write_text(json.dumps(indices))
            created = []
            factory = SerializerFactory.create_serializer

            def observe(*args, **kwargs):
                serializer = factory(*args, **kwargs)
                created.append(serializer)
                return serializer

            with patch.object(config_module, "ProjectConfig", return_value=config), patch.object(
                    SerializerFactory, "create_serializer", side_effect=observe):
                _, result = _prepare_one_mp((dataset, method, 8, 2, 1, "fit_scope_oracle", "tests", 1, False))
            assert result["success"], result
            udi = UnifiedDataInterface(config, dataset)
            loader = udi.get_dataset_loader()
            all_data, _ = loader.get_all_data_with_indices()
            assert [int(s["id"].split("_")[-1]) for s in all_data] == fit_order
            serialized = [seq for _, seq in udi.get_sequences(method)[0]]
            codebook = udi.get_bpe_codebook(method)
            engine = udi.get_bpe_encoder(method, encode_rank_mode="all")
            vocab = udi.get_vocab(method)
            inputs = serialized if mode == "none" else engine.batch_encode(serialized)
            serializer = created[0]
            frequencies = []
            # Numeric sparse keys encode (source type, edge type, target type).
            if method == "feuler":
                _, e_dim, v_dim = serializer._triplet_tensor_dims
                frequencies = sorted([[list((int(k) // (e_dim * v_dim),
                                              (int(k) // v_dim) % e_dim, int(k) % v_dim)), int(v)]
                                      for k, v in serializer._triplet_sparse_counts.items()])
                if dataset == "aqsol":
                    # Native AQSOL stores already encoded atomic/bond tokens.
                    frequencies = [[[2 * a + 1, 2 * e, 2 * b + 1], n]
                                   for (a, e, b), n in frequencies]
            cases.append(dict(dataset=dataset, encoder=encoder, fit_order=fit_order,
                              graphs=graphs, serialized=serialized, merge_rules=codebook["merge_rules"],
                              vocabulary={str(token): index for token, index in vocab.token_to_id.items() if token >= 0},
                              input_ids=[vocab.encode_sequence(seq)["input_ids"].tolist() for seq in inputs],
                              frequencies=frequencies))
    result = dict(repository="BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers",
                  revision=REVISION, python=platform.python_version(), torch=torch.__version__,
                  dgl=dgl.__version__, split_indices=split_indices,
                  source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sorted(set(sources))},
                  cases=cases)
    destination = Path(__file__).with_name("graph_tokenizer_fit_scope_official.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(destination)
