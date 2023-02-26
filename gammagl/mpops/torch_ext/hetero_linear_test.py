import torch
from torch_hetero_linear import segment_matmul
import time

num_types = 10000
num_nodes = 1000
hidden_dim = 64

inputs = torch.randn(num_nodes * num_types, hidden_dim)
ptr = torch.arange(num_types+1) * num_nodes
other = torch.randn((num_types, hidden_dim, hidden_dim), requires_grad=True)

def test_ext():
    t0 = time.time()
    segment_matmul(inputs, ptr, other)
    print(time.time() - t0)

def test_torch():
    t0 = time.time()
    for i in range(num_types):
        inp = inputs[num_nodes*i:num_nodes*(i+1)]
        oth = other[i]
        torch.matmul(inp, oth)
    print(time.time() - t0)


test_ext()
test_torch()