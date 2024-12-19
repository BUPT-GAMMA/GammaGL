import os
# os.environ['TL_BACKEND'] = "torch"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorlayerx as tlx
from gammagl.mpops import *
import numpy as np
import time

try:
    tlx.set_device("GPU", 0)
except:
    print("GPU is not available")

relative_path = '/home/zgy/review/zgy/GammaGL/profiler/mpops/edge_index/'
file_name = ['cora.npy', 'pubmed.npy', 'ogbn-arxiv.npy']
embedding = [16, 64, 256]
iter = 1000

operations = {
    'segment_max': segment_max,
    'unsorted_segmnet_max': unsorted_segment_max,
    'segment_sum': segment_sum,
    'unsorted_segment_sum': unsorted_segment_sum,
    'segment_mean': segment_mean,
    'unsorted_segment_mean': unsorted_segment_mean
}


def run_test(operation, msg, dst, num_nodes):
    operation(msg, dst, num_nodes) # warm up
    start = time.time()
    for j in range(iter):
        operation(msg, dst, num_nodes)
    end = time.time()
    return end - start


with open('test_results.txt', 'w') as result_file:
    for name in file_name:
        path = relative_path + name
        result_file.write(f"Loading data from {path}\n")
        print(f"Loading data from {path}")
        edge_index = np.load(path)

        num_nodes = np.max(edge_index) + 1
        src = tlx.convert_to_tensor(edge_index[0, :], tlx.int64)
        dst = tlx.convert_to_tensor(edge_index[1, :], tlx.int64)
        dst2 = None

        # TensorFlow-specific preprocessing for sorted operations
        if tlx.BACKEND == 'tensorflow' or tlx.BACKEND == 'paddle' or tlx.BACKEND == 'mindspore':
            dst_numpy = tlx.convert_to_numpy(dst)
            idx = np.argsort(dst_numpy)
            dst2 = tlx.gather(tlx.convert_to_tensor(dst_numpy, dtype=tlx.int64), tlx.convert_to_tensor(idx, dtype=tlx.int64))

        for embedding_dim in embedding:
            result_file.write(f"** Testing embedding dimension {embedding_dim} **\n")
            print(f"** Testing embedding dimension {embedding_dim} **")
            x = tlx.convert_to_tensor(np.random.randn(num_nodes, embedding_dim), dtype=tlx.float32)
            msg = tlx.gather(x, src)

            # Running tests for all operations
            for op_name, op_func in operations.items():
                dst_target = dst2 if dst2 is not None else dst
                duration = run_test(op_func, msg, dst_target, num_nodes)
                result = f"{op_name}: {duration:.3f} seconds\n"
                result_file.write(result)
                print(result)

            print(f"** Done testing embedding dimension {embedding_dim} **")
            result_file.write(f"** Done testing embedding dimension {embedding_dim} **\n")

        if tlx.BACKEND == 'paddle':
            info = f"Data tensors are on device: {x.place}\n"
        elif tlx.BACKEND == 'mindspore':
            info = f"mindspore\n"
        else:
            info = f"Data tensors are on device: {x.device}\n"
        print(info, end='')
        result_file.write(info)

