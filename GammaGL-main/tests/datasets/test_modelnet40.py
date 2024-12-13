from gammagl.datasets import ModelNet40

root = './data'
def test_modelnet40(get_dataset):
    train_dataset = get_dataset(name = 'ModelNet40')
    assert train_dataset.num_features == 3
    assert train_dataset.num_node_features == 3
    assert train_dataset.num_points == 1024
    assert train_dataset.split == 'train'
    test_dataset = get_dataset(name = 'ModelNet40',split = 'test')
    assert test_dataset.num_features == 3
    assert test_dataset.num_node_features == 3
    assert test_dataset.num_points == 1024
    assert test_dataset.split == 'test'
