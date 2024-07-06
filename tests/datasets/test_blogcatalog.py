from gammagl.data import InMemoryDataset, download_url, Graph
import unittest
from gammagl.datasets.blogcatalog import BlogCatalog

def test_blogcatalog():
    dataset = BlogCatalog(root='./temp')
    graph = dataset[0]
    assert isinstance(graph, Graph)
    assert graph.num_nodes == 5106
    assert graph.num_edges == 171743
    assert graph.num_features == 8189
    assert graph.num_classes == 6
    assert graph.train_mask.sum().item() == 2553
    assert graph.val_mask.sum().item() == 1276
    assert graph.test_mask.sum().item() == 1277
