from gammagl.datasets.modelnet40 import ModelNet40

root = './data'
def test_modelnet40_dataset_train():
    train_dataset = ModelNet40(root)
    assert train_dataset.num_features == 3
    assert train_dataset.num_node_features == 3
    assert train_dataset.num_points == 1024
    assert train_dataset.split == 'train'
def test_modelnet40_dataset_test():
    test_dataset = ModelNet40(root,split = 'test')
    assert test_dataset.num_features == 3
    assert test_dataset.num_node_features == 3
    assert test_dataset.num_points == 1024
    assert test_dataset.split == 'test'
