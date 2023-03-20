import numpy as np


def test_imdb(get_dataset):
    dataset = get_dataset(name='IMDB-BINARY')
    assert len(dataset) == 1000
    assert dataset.num_features == 0
    assert dataset.num_classes == 2
    assert str(dataset) == 'IMDB-BINARY(1000)'

    data = dataset[0]
    print(data)
    assert len(data) == 2
    assert data.edge_index.shape == (2, 146)
    assert data.y.shape == (1,)
    assert data.num_nodes == 20

