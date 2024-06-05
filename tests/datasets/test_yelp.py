import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['TL_BACKEND'] = 'paddle'
import tensorlayerx as tlx
from gammagl.datasets import Yelp

def yelp():
    dataset = Yelp()
    graph = dataset[0]
    assert len(dataset) == 1
    assert dataset.num_classes == 100
    assert dataset.num_node_features == 300
    assert graph.edge_index.shape[1] == 13954819
    assert graph.x.shape[0] == 716847
