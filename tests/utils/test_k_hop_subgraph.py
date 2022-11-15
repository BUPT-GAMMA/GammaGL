import tensorlayerx as tlx
from gammagl.utils import k_hop_subgraph

def test_k_hop_subgraph():
    edge_index = tlx.convert_to_tensor([[0, 1, 2, 3, 4, 5], [2, 2, 4, 4, 6, 6]])
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(6, 2, edge_index, relabel_nodes=True)
    assert subset.numpy().tolist() == [2, 3, 4, 5, 6]
    assert edge_index.numpy().tolist() == [[0, 1, 2, 3], [2, 2, 4, 4]]
    assert mapping.tolist() == [4]
    assert edge_mask.numpy().tolist() == [False, False,  True,  True,  True,  True]

    edge_index = tlx.convert_to_tensor([[1, 2, 4, 5], [0, 1, 5, 6]])
    subset, edge_index, mapping, edge_mask = k_hop_subgraph([0, 6], 2, edge_index, relabel_nodes=True)
    assert subset.numpy().tolist() == [0, 1, 2, 4, 5, 6]
    assert edge_index.numpy().tolist() == [[1, 2, 3, 4], [0, 1, 4, 5]]
    assert mapping.tolist() == [0, 5]
    assert edge_mask.numpy().tolist() == [True,  True,  True,  True]