from time import time

import tensorlayerx as tlx
from gammagl.datasets import Reddit
from gammagl.loader.neighbor_sampler import NeighborSampler


def test_reddit():
    aaa = time()
    # dataset = Reddit('./reddit', 'reddit')
    dataset = Reddit()
    g = dataset.data
    bbb = time()
    print(bbb - aaa)
    aaa = time()
    test_sample(g)
    bbb = time()
    print(bbb - aaa)
    pass


def test_sample(graph):
    loader = NeighborSampler(edge_index=graph.edge_index,
                             node_idx=tlx.arange(10000),
                             sample_lists=[25, 10], batch_size=2048, shuffle=True, num_workers=0)
    for dst_node, adjs, all_node in loader:
        pass
