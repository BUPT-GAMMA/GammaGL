import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import sys
sys.path.insert(0, '/home/zbs2/GammaGL_algo/论文复现/GraphLama/GammaGL/tensorlayerx_patch')

import gammagl
import torch
from gammagl.data import Graph

print("GammaGL version:", getattr(gammagl, '__version__', 'Unknown'))
print("PyTorch version:", torch.__version__)

# Create a simple graph
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.randn(3, 4)
graph = Graph(x=x, edge_index=edge_index)

print("Graph created successfully:")
print(f"  Nodes: {graph.num_nodes}")
print(f"  Edges: {graph.num_edges}")
print(f"  Node features shape: {graph.x.shape}")
print("\nGammaGL basic functionality test passed!")
