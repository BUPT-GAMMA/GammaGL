import time
import torch
import numpy as np

from pyinstrument import Profiler
pf = Profiler()

# copied from gammagl/mpops/torch.py
def unsorted_segment_max(x, segment_ids, num_segments=None):
    if num_segments is None:
        num_segments = int(segment_ids.max().item()) + 1

    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    res = []
    for i in range(num_segments):
        res.append(torch.max(x[segment_ids == i], dim=0)[0])
    return torch.stack(res, dim=0)

def unsorted_segment_sum(x, segment_ids, num_segments=None):
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    if num_segments is None:
        num_segments = int(segment_ids.max().item()) + 1
    # else:
        # `rgcn` meet an error that `segment_ids` is empty
        # assert segment_ids.max() < num_segments


    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(x.shape[1:], device=x.device)).to(torch.int64)
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *x.shape[1:])

    assert x.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(x.shape[1:])
    tensor = torch.zeros(*shape, device=x.device).to(x.dtype).scatter_add(0, segment_ids, x)
    return tensor

try:
    import torch_segment
except ImportError:
    exit(0)

edge_index = np.load('edge_index.npy')
num_nodes = np.max(edge_index)+1
src = edge_index[0,:]
dst = edge_index[1,:]
src = torch.from_numpy(src)
dst = torch.from_numpy(dst)
x = torch.from_numpy(np.random.randn(num_nodes, 256)).float()

py_iter = 10
ext_iter = 10 # about 10~80 times acceleration
# The time consuming of `unsorted_segment_sum` is mainly 
# the `for` loop, which is related to the number of edges. 
# However, the time-consuming of `torch_segment.segment_max`
# is determined by the product of the number of edges 
# and the feature dimension.
pf.start()
for j in range(ext_iter):
    msg = x[src]
    torch_segment.segment_max(msg, dst, num_nodes)
pf.stop()
print(pf.output_text(unicode=True, color=True))

pf.start()
for j in range(py_iter):
    msg = x[src]
    unsorted_segment_max(msg, dst, num_nodes)
pf.stop()
print(pf.output_text(unicode=True, color=True))

# msg = torch.randn((edge_index.shape[1], 100))
# start_t = time.time()
# for j in range(5):
#     unsorted_segment_max_cpp(msg, dst, num_nodes)
# print("{:.3f}".format(time.time()-start_t))

# start_t = time.time()
# for j in range(5):
#     unsorted_segment_max(msg, dst, num_nodes)
# print("{:.3f}".format(time.time()-start_t))


pf.start()
for j in range(py_iter):
    msg = x[src]
    unsorted_segment_sum(msg, dst, num_nodes)
pf.stop()
print(pf.output_text(unicode=True, color=True))


try:
    import torch_gspmm
except ImportError:
    exit(0)
edge_index = torch.tensor(edge_index, dtype=torch.int64)
weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

pf.start()
for j in range(ext_iter):
    torch_gspmm.spmm_sum(edge_index, weight, x)
pf.stop()
print(pf.output_text(unicode=True, color=True))
