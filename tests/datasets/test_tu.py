
def test_ENZYMES(get_dataset):
    dataset = get_dataset(name='ENZYMES')
    assert len(dataset) == 600
    assert dataset.num_classes == 6
    assert dataset.num_node_features == 3

