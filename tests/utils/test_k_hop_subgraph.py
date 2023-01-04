import tensorlayerx as tlx
from gammagl.utils import k_hop_subgraph

def test_k_hop_subgraph():
    edge_index = tlx.convert_to_tensor([[0, 1, 2, 3, 4, 5], [2, 2, 4, 4, 6, 6]], dtype = tlx.int64)
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(6, 2, edge_index, relabel_nodes=True)
    assert tlx.convert_to_numpy(subset).tolist() == [2, 3, 4, 5, 6]
    assert tlx.convert_to_numpy(edge_index).tolist() == [[0, 1, 2, 3], [2, 2, 4, 4]]
    assert mapping.tolist() == [4]
    assert tlx.convert_to_numpy(edge_mask).tolist() == [False, False,  True,  True,  True,  True]

    edge_index = tlx.convert_to_tensor([[1, 2, 4, 5], [0, 1, 5, 6]])
    subset, edge_index, mapping, edge_mask = k_hop_subgraph([0, 6], 2, edge_index, relabel_nodes=True)
    assert tlx.convert_to_numpy(subset).tolist() == [0, 1, 2, 4, 5, 6]
    assert tlx.convert_to_numpy(edge_index).tolist() == [[1, 2, 3, 4], [0, 1, 4, 5]]
    assert mapping.tolist() == [0, 5]
    assert tlx.convert_to_numpy(edge_mask).tolist() == [True,  True,  True,  True]
