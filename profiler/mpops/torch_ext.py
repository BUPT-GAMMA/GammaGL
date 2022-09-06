import time
import torch
import numpy as np

# copied from gammagl/mpops/torch.py
def unsorted_segment_max(x, segment_ids, num_segments=None):
    if num_segments is None:
        num_segments = int(segment_ids.max().item()) + 1

    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    res = []
    for i in range(num_segments):
        res.append(torch.max(x[segment_ids == i], dim=0)[0])
    return torch.stack(res, dim=0)

torch.ops.load_library('../../gammagl/mpops/torch_segment/build/libtorch_segment.so')
unsorted_segment_max_cpp = torch.ops.mp.segment_max

edge_index = np.load('edge_index.npy')
num_nodes = np.max(edge_index)+1
src = edge_index[0,:]
dst = edge_index[1,:]
src = torch.from_numpy(src)
dst = torch.from_numpy(dst)
x = torch.from_numpy(np.random.randn(num_nodes, 100))

start_t = time.time()
for j in range(5):
    msg = x[src]
    unsorted_segment_max_cpp(msg, dst, num_nodes)
print("{:.3f}".format(time.time()-start_t))

start_t = time.time()
for j in range(5):
    msg = x[src]
    unsorted_segment_max(msg, dst, num_nodes)
print("{:.3f}".format(time.time()-start_t))


# msg = torch.randn((edge_index.shape[1], 100))
# start_t = time.time()
# for j in range(5):
#     unsorted_segment_max_cpp(msg, dst, num_nodes)
# print("{:.3f}".format(time.time()-start_t))

# start_t = time.time()
# for j in range(5):
#     unsorted_segment_max(msg, dst, num_nodes)
# print("{:.3f}".format(time.time()-start_t))
