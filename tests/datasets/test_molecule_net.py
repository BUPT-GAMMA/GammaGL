import tensorlayerx as tlx
from gammagl.utils.smiles import from_smiles
from gammagl.datasets.molecule_net import MoleculeNet


def test_moleculenet():
    root = './temp'
    dataset = MoleculeNet(root=root, name='ESOL')
    data = dataset[0]
    assert data.y.shape[1] == 1, "Label shape mismatch"
    assert len(data.x) > 0, "Node features should not be empty"
    assert data.edge_index.shape[0] == 2, "Edge index shape mismatch"
    print("All tests passed!")




