from gammagl.datasets.ppi import PPI


def test_ppi_dataset():
    train_dataset = PPI()
    val_dataset = PPI(split='val')
    test_dataset = PPI(split='test')
    assert len(train_dataset) == 20
    assert len(val_dataset) == 2
    assert len(test_dataset) == 2

