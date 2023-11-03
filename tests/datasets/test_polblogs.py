import tensorlayerx as tlx
from gammagl.datasets.polblogs import PolBlogs

root = "./data"

def test_polblogs_dataset():
    polblogs = PolBlogs(root)
    assert len(polblogs) == 1
    assert polblogs.num_classes == 2
    assert polblogs.num_node_features == 1490
    edge_num = tlx.get_tensor_shape(polblogs.data["edge_index"])[1]
    assert edge_num ==19025
    assert polblogs.num_features == 1490
    node_num = tlx.get_tensor_shape(polblogs.data["x"])[0]
    assert node_num == 1490

test_polblogs_dataset()