import numpy as np
import tensorlayerx as tlx
import scipy.sparse as sp
from gammagl.layers.conv import HEATlayer


def test_heat_layer():
    x = tlx.random_normal(shape=(4, 64))
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_attrs = tlx.random_normal(shape=(6, 5))
    edge_types = tlx.convert_to_tensor(np.random.randint(0, 4, size=(6, 4)))

    conv = HEATlayer(in_channels_node=64, in_channels_edge_attr=5, in_channels_edge_type=4,
                     node_emb_size=64, edge_attr_emb_size=64, edge_type_emb_size=64, out_channels=128, heads=3, concat=True)

    out = conv(x, edge_index, edge_attrs, edge_types)

    assert tlx.get_tensor_shape(out) == [4, 128]


# if __name__ == "__main__":
#     test_heat_layer()


