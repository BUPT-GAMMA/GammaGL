from gammagl.data import Graph
from gammagl.datasets.blogcatalog import BlogCatalog
import tensorlayerx as tlx


def test_blogcatalog():
    dataset = BlogCatalog(root='./temp')
    graph = dataset[0]
    assert isinstance(graph, Graph)
    assert graph.num_nodes == 5196
    assert graph.num_edges == 343486
    assert graph.num_features == 8189
    assert dataset.num_classes == 6
    assert tlx.reduce_sum(tlx.cast(graph.train_mask, dtype=tlx.int64)) == 2598
    assert tlx.reduce_sum(tlx.cast(graph.val_mask, dtype=tlx.int64)) == 1299
    assert tlx.reduce_sum(tlx.cast(graph.test_mask, dtype=tlx.int64)) == 1299
