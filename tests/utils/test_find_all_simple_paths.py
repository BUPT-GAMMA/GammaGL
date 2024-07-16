import tensorlayerx as tlx
from gammagl.utils.simple_path import find_all_simple_paths


def test_find_all_simple_paths():
    if tlx.BACKEND == 'tensorflow':
        return
    edge_index = tlx.convert_to_tensor([[0, 0, 1, 1, 2], [1, 2, 0, 3, 1]], dtype=tlx.int64)
    src = tlx.convert_to_tensor([0], dtype=tlx.int64)
    dest = tlx.convert_to_tensor([3], dtype=tlx.int64)
    expected_paths = [[0, 1, 3], [0, 2, 1, 3]]
    paths = find_all_simple_paths(edge_index, src, dest, max_length=4)
    assert sorted(map(tuple, paths))==sorted(map(tuple, expected_paths))

    # no path
    src = tlx.convert_to_tensor([0], dtype=tlx.int64)
    dest = tlx.convert_to_tensor([4], dtype=tlx.int64)
    expected_paths = []
    paths = find_all_simple_paths(edge_index, src, dest, max_length=4)
    assert sorted(map(tuple, paths)) == sorted(map(tuple, expected_paths))

    # same start and end point
    dest = tlx.convert_to_tensor([0], dtype=tlx.int64)
    expected_paths = [[0]]
    paths = find_all_simple_paths(edge_index, src, dest, max_length=1)
    assert sorted(map(tuple, paths))==sorted(map(tuple, expected_paths))
