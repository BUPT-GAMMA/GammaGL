from gammagl.utils import negative_sampling
import tensorlayerx as tlx
import numpy as np

def not_negative(edge_index, neg_edge_index, size, bipartite):
    a = tlx.convert_to_numpy(edge_index).T.tolist()
    b = tlx.convert_to_numpy(neg_edge_index).T.tolist()
    if not bipartite:   
        c = np.array([list(range(size[0])),list(range(size[0]))]).T.tolist()
        assert not any(x in b for x in c)
    
    return not any(x in b for x in a)

def test_negative_sampling():
    edge_index = tlx.convert_to_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
    neg_edge_index = negative_sampling(edge_index)
    assert tlx.get_tensor_shape(neg_edge_index)[1] == tlx.get_tensor_shape(edge_index)[1]
    assert not_negative(edge_index, neg_edge_index, (4, 4), bipartite=False)

    neg_edge_index = negative_sampling(edge_index, method='dense', num_neg_samples=3)
    assert tlx.get_tensor_shape(neg_edge_index)[1] == 3
    assert not_negative(edge_index, neg_edge_index, (4, 4), bipartite=False)
