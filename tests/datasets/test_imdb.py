import numpy as np
import tensorlayerx as tlx

def test_imdb(get_dataset):
    dataset = get_dataset(name = 'IMDB-BINARY')
    assert len(dataset) == 1000
    assert dataset.num_features == 0
    assert dataset.num_classes == 2
    assert str(dataset) == 'IMDB-BINARY(1000)'

    data = dataset[0]
    assert len(data) == 2
    assert tlx.get_tensor_shape(data.edge_index) == [2, 146]
    assert tlx.get_tensor_shape(data.y) == [1,]
    assert data.num_nodes == 20

