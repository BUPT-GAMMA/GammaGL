import tensorlayerx as tlx
from gammagl.datasets.ml import MLDataset  # Replace with the correct module path


def test_mldataset():
    if tlx.BACKEND == "tensorflow":
        return
    root = './temp'
    dataset = MLDataset(root=root, dataset_name='ml-100k')
    data = dataset[0]
    assert data.edge_index.shape[0] == 2, "Edge index shape mismatch"
    assert len(data.edge_weight) > 0, "Edge weights should not be empty"
    assert len(data.user_id) > 0, "User IDs should not be empty"
    assert len(data.item_id) > 0, "Item IDs should not be empty"
    print("All tests passed!")
