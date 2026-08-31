import ast
import importlib
from pathlib import Path

import pytest


def test_graph_bpe_setup_targets_runtime_module_name():
    setup_path = (
        Path(__file__).resolve().parents[2]
        / "third_party"
        / "graph_bpe_cpp"
        / "setup.py"
    )
    tree = ast.parse(setup_path.read_text(encoding="utf-8"))
    extension_names = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Extension"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]

    assert extension_names == ["third_party.graph_bpe_cpp._graph_bpe"]


def test_built_graph_bpe_extension_matches_python_reference():
    native = pytest.importorskip("third_party.graph_bpe_cpp._graph_bpe")
    bridge = importlib.import_module("third_party.graph_bpe_cpp")
    sequences = [[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 3]]

    native_result = native.train_bpe(sequences, 2, 2)
    reference = bridge._train_bpe_python(sequences, 2, 2)

    assert native_result["merge_rules"] == reference["merge_rules"]
    assert native_result["vocab_size"] == reference["vocab_size"]
