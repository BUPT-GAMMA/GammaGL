import numpy as np
import tensorlayerx as tlx
from gammagl.layers.pool.glob import global_sum_pool, global_mean_pool, global_max_pool, global_min_pool, global_sort_pool

def test_glob():
    N = 10  
    F = 5   
    B = 2  
    x = tlx.convert_to_tensor(np.random.randn(N * B, F), dtype=tlx.float32)
    batch = tlx.convert_to_tensor(np.repeat(np.arange(B), N), dtype=tlx.int64)
    sum_pooled = global_sum_pool(x, batch)
    assert sum_pooled.shape == (B, F), "Global sum pool output shape mismatch"
    assert np.allclose(sum_pooled.numpy(), np.array([x.numpy()[:N].sum(axis=0), x.numpy()[N:].sum(axis=0)])), "Global sum pool output values mismatch"
    mean_pooled = global_mean_pool(x, batch)
    assert mean_pooled.shape == (B, F), "Global mean pool output shape mismatch"
    assert np.allclose(mean_pooled.numpy(), np.array([x.numpy()[:N].mean(axis=0), x.numpy()[N:].mean(axis=0)])), "Global mean pool output values mismatch"
    max_pooled = global_max_pool(x, batch)
    assert max_pooled.shape == (B, F), "Global max pool output shape mismatch"
    assert np.allclose(max_pooled.numpy(), np.array([x.numpy()[:N].max(axis=0), x.numpy()[N:].max(axis=0)])), "Global max pool output values mismatch"
    min_pooled = global_min_pool(x, batch)
    assert min_pooled.shape == (B, F), "Global min pool output shape mismatch"
    assert np.allclose(min_pooled.numpy(), np.array([x.numpy()[:N].min(axis=0), x.numpy()[N:].min(axis=0)])), "Global min pool output values mismatch"
    k = 3
    sort_pooled = global_sort_pool(x, batch, k)
    assert sort_pooled.shape == (B, k * F), f"Global sort pool output shape mismatch: {sort_pooled.shape} != ({B}, {k * F})"
    sorted_x1 = x.numpy()[:N][np.argsort(x.numpy()[:N][:, -1])[::-1]]
    sorted_x2 = x.numpy()[N:][np.argsort(x.numpy()[N:][:, -1])[::-1]]
    sorted_x_combined = np.vstack([sorted_x1, sorted_x2])
    for b in range(B):
        assert np.allclose(sort_pooled[b, :k*F], sorted_x_combined[b*N:b*N+k, :].flatten()), f"Global sort pool output values mismatch for graph {b+1}"


