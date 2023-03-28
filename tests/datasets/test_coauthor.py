import sys
import os
import tensorlayerx as tlx
from gammagl.datasets.coauthor import Coauthor

root = "./data"
def test_coauthor_dataset():
    dataset_CS = Coauthor(root, name = "CS")
    assert len(dataset_CS) == 1
    assert dataset_CS.num_classes == 15
    assert dataset_CS.num_node_features == 6805
    assert dataset_CS.num_features == 6805
    edge_num = tlx.get_tensor_shape(dataset_CS.data["edge_index"])[1]
    assert edge_num == 163788
    node_num = tlx.get_tensor_shape(dataset_CS.data["x"])[0]
    assert node_num == 18333

    dataset_Physics = Coauthor(root, name = "Physics")
    assert len(dataset_Physics) == 1
    assert dataset_Physics.num_classes == 5
    assert dataset_Physics.num_node_features == 8415
    assert dataset_Physics.num_features == 8415
    edge_num = tlx.get_tensor_shape(dataset_Physics.data["edge_index"])[1]
    assert edge_num == 495924
    node_num = tlx.get_tensor_shape(dataset_Physics.data["x"])[0]
    assert node_num == 34493
