import pickle
import sys
import tempfile

import dgl
import torch
import torchdata


assert sys.version_info[:2] == (3, 10)
assert torch.__version__.split("+")[0] == "2.1.2"
assert dgl.__version__ == "2.1.0"
assert torchdata.__version__ == "0.7.1"

graph = dgl.graph(([0], [1]), num_nodes=2)
graph.ndata["node_token_ids"] = torch.tensor([[5], [9]])

with tempfile.TemporaryFile() as handle:
    pickle.dump([(graph, 1)], handle)
    handle.seek(0)
    restored = pickle.load(handle)[0][0]

assert tuple(values.tolist() for values in restored.edges()) == ([0], [1])
assert restored.ndata["node_token_ids"].tolist() == [[5], [9]]
