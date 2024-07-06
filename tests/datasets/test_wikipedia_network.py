from gammagl.datasets.wikipedia_network import  WikipediaNetwork


def test_wikipedia_network():
    root = './temp'
    dataset = WikipediaNetwork(root=root, name='chameleon', geom_gcn_preprocess=True)
    assert len(dataset) > 0, "Dataset should not be empty"
    data = dataset[0]
    assert data.x.shape[0] > 0, "Node features should not be empty"
    assert data.edge_index.shape[0] == 2, "Edge index shape mismatch"
    assert data.y.shape[0] == data.x.shape[0], "Labels shape mismatch"
    assert data.train_mask.shape[0] == data.x.shape[0], "Train mask shape mismatch"
    assert data.val_mask.shape[0] == data.x.shape[0], "Validation mask shape mismatch"
    assert data.test_mask.shape[0] == data.x.shape[0], "Test mask shape mismatch"
    print("All tests passed!")
