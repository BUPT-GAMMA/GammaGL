import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['TL_BACKEND'] = 'paddle'
import tensorlayerx as tlx
from gammagl.datasets.aminer import AMiner

root = "./data"
def test_aminer():
    dataset = AMiner(root)
    graph = dataset[0]
    assert len(dataset) == 1
    assert graph['author'].num_features == 0
    assert graph['venue'].num_features == 0
    assert graph['paper'].num_features == 0

    assert graph.num_edges == 25036020
    assert tlx.get_tensor_shape(graph[('paper', 'written_by', 'author')].edge_index) == [2, 9323605]
    assert tlx.get_tensor_shape(graph[('venue', 'publishes', 'paper')].edge_index) == [2, 3194405]
    assert tlx.get_tensor_shape(graph[('paper', 'published_in', 'venue')].edge_index) == [2, 3194405]
    assert tlx.get_tensor_shape(graph[('author', 'writes', 'paper')].edge_index) == [2, 9323605]

    assert graph['author'].num_nodes == 1693531
    assert graph['venue'].num_nodes == 3883
    assert graph['paper'].num_nodes == 3194405
    assert graph.num_nodes == 4891819


test_aminer()
