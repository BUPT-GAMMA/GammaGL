import pytest
import tensorlayerx as tlx
from gammagl.utils import homophily


def test_homophily():
    edge_index = tlx.convert_to_tensor(([[0, 1, 2, 3], [1, 2, 0, 4]]))
    y = tlx.convert_to_tensor(([0, 0, 0, 0, 1]))
    batch = tlx.convert_to_tensor(([0, 0, 0, 1, 1]))

    method = 'edge'
    assert pytest.approx(homophily(edge_index, y, method=method)) == 0.75
    # assert pytest.approx(homophily(adj, y, method=method)) == 0.75
    assert pytest.approx(tlx.convert_to_numpy(homophily(edge_index, y, batch, method)[0])) == 1.0

    method = 'node'
    assert pytest.approx(homophily(edge_index, y, method=method)) == 0.6
    # assert pytest.approx(homophily(adj, y, method=method)) == 0.6
    assert pytest.approx(tlx.convert_to_numpy(homophily(edge_index, y, batch, method)[0])) == 1.0

    method = 'edge_insensitive'
    assert pytest.approx(homophily(edge_index, y, method=method)) == 0.1999999
    # assert pytest.approx(homophily(adj, y, method=method)) == 0.1999999
    assert pytest.approx(homophily(edge_index, y, batch, method)[0]) == 0.0
