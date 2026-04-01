import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import sys
sys.path.insert(0, '/home/zbs2/GammaGL_algo/论文复现/GraphLama/GammaGL/tensorlayerx_patch')

import gammagl
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.utils.check import check_is_numpy

print("Loading Cora dataset...")
dataset = Planetoid(root='/tmp/cora', name='cora')
graph = dataset[0]

print(f"Edge index type: {type(graph.edge_index)}")
print(f"Edge index shape: {graph.edge_index.shape}")
print(f"Is tensor: {tlx.is_tensor(graph.edge_index)}")
print(f"Is numpy: {check_is_numpy(graph.edge_index)}")
print(f"Edge index dtype: {graph.edge_index.dtype}")
print(f"First few edges: {graph.edge_index[:, :5]}")
