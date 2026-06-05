import os
os.environ['TL_BACKEND'] = 'torch'

def test_qm9_dataset_import():
    from gammagl.datasets.qm9_dataset import QM9Gen
    assert QM9Gen is not None

def test_guacamol_dataset_import():
    from gammagl.datasets.guacamol_dataset import GuacaMolDataset
    assert GuacaMolDataset is not None

def test_moses_dataset_import():
    from gammagl.datasets.moses_dataset import MOSESDataset
    assert MOSESDataset is not None

def test_zinc250k_dataset_import():
    from gammagl.datasets.zinc250k_dataset import ZINC250kGen
    assert ZINC250kGen is not None

def test_spectre_dataset_import():
    from gammagl.datasets.spectre_dataset import PlanarGraphDataset, SBMGraphDataset
    assert PlanarGraphDataset is not None
    assert SBMGraphDataset is not None

def test_tls_dataset_import():
    from gammagl.datasets.tls_dataset import TLSGraphDataset
    assert TLSGraphDataset is not None

if __name__ == '__main__':
    test_qm9_dataset_import()
    test_guacamol_dataset_import()
    test_moses_dataset_import()
    test_zinc250k_dataset_import()
    test_spectre_dataset_import()
    test_tls_dataset_import()
    print("Dataset import tests passed!")
