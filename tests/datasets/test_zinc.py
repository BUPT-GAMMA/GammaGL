from gammagl.datasets.zinc import ZINC


def test_zinc():
    root = './temp'
    dataset = ZINC(root=root, subset=False, split='train')
    assert len(dataset) > 0, "Dataset should not be empty"
    data = dataset[0]
    assert data.x.shape[0] > 0, "Node features should not be empty"
    assert data.edge_index.shape[0] == 2, "Edge index shape mismatch"
    assert data.edge_attr.shape[0] == data.edge_index.shape[1], "Edge attributes shape mismatch"
    # assert data.y.shape[0] == data.x.shape[0], "Labels shape mismatch"
    print("All tests passed!")
