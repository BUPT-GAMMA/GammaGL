import tensorlayerx as tlx
from gammagl.datasets.wikics import WikiCS

root = "./data"

def test_wikics_dataset():
    wikics = WikiCS(root)
    assert len(wikics) == 1
    assert wikics.num_classes == 10
    assert wikics.num_node_features == 300
    assert wikics.num_features == 300
    node_num = tlx.get_tensor_shape(wikics.data["x"])[0]
    assert node_num == 11701