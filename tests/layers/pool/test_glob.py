import numpy as np
import tensorlayerx as tlx
from gammagl.layers.pool.glob  import global_sum_pool, global_mean_pool, global_max_pool, global_min_pool, global_sort_pool
def test_glob():
    # Example data
    N = 10  # Number of nodes
    F = 5   # Number of features per node
    B = 2   # Number of graphs

    # Node features
    x = tlx.convert_to_tensor(np.random.randn(N * B, F), dtype=tlx.float32)

    # Batch vector
    batch = tlx.convert_to_tensor(np.repeat(np.arange(B), N), dtype=tlx.int64)

    # Test global sum pool
    sum_pooled = global_sum_pool(x, batch)
    print("Global sum pool result:", sum_pooled.numpy())

    # Test global mean pool
    mean_pooled = global_mean_pool(x, batch)
    print("Global mean pool result:", mean_pooled.numpy())

    # Test global max pool
    max_pooled = global_max_pool(x, batch)
    print("Global max pool result:", max_pooled.numpy())

    # Test global min pool
    min_pooled = global_min_pool(x, batch)
    print("Global min pool result:", min_pooled.numpy())

    # Test global sort pool
    k = 3
    sort_pooled = global_sort_pool(x, batch, k)
    print(f"Global sort pool result (top {k} nodes):", sort_pooled.numpy())
