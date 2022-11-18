from gammagl.utils import negative_sampling
import tensorlayerx as tlx

def test_negative_sampling():
    tlx.set_device()
    edge_index = tlx.convert_to_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
    print('graph:')
    print(negative_sampling(edge_index))
    print('bipartite graph:')
    print(negative_sampling(edge_index, num_nodes = (3, 4)))