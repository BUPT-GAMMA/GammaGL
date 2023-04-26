import sys
import os
import tensorlayerx as tlx
from gammagl.datasets.amazon import Amazon

root = "./data"
def test_amazon_dataset():
    dataset_computers = Amazon(root, name = "Computers")
    assert len(dataset_computers) == 1
    assert dataset_computers.num_classes == 10
    assert dataset_computers.num_node_features == 767
    assert dataset_computers.num_features == 767
    edge_num = tlx.get_tensor_shape(dataset_computers.data["edge_index"])[1]
    assert edge_num == 491722
    node_num = tlx.get_tensor_shape(dataset_computers.data["x"])[0]
    assert node_num == 13752

    dataset_photo = Amazon(root, name = "Photo")
    assert len(dataset_photo) == 1
    assert dataset_photo.num_classes == 8
    assert dataset_photo.num_node_features == 745
    assert dataset_photo.num_features == 745
    edge_num = tlx.get_tensor_shape(dataset_photo.data["edge_index"])[1]
    assert edge_num == 238162
    node_num = tlx.get_tensor_shape(dataset_photo.data["x"])[0]
    assert node_num == 7650
