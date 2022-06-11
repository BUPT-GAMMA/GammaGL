import tensorlayerx as tlx
from gammagl.utils import degree


def test_degree():
	row = tlx.convert_to_tensor([0, 1, 0, 2, 0])
	deg = degree(row, dtype=tlx.int64)
	assert deg.dtype == tlx.int64
	assert tlx.convert_to_numpy(deg).tolist() == [3, 1, 1]

