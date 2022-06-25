import tensorlayerx as tlx

from gammagl.utils import (
    # add_remaining_self_loops,
    add_self_loops,
    contains_self_loops,
    # get_self_loop_attr,
    remove_self_loops,
    # segregate_self_loops,
)


def test_contains_self_loops():
    edge_index = tlx.convert_to_tensor([[0, 1, 0], [1, 0, 0]])
    assert contains_self_loops(edge_index)

    edge_index = tlx.convert_to_tensor([[0, 1, 1], [1, 0, 2]])
    assert not contains_self_loops(edge_index)


def test_remove_self_loops():
    edge_index = tlx.convert_to_tensor([[0, 1, 0], [1, 0, 0]])
    edge_attr = [[1, 2], [3, 4], [5, 6]]
    edge_attr = tlx.convert_to_tensor(edge_attr)

    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    assert tlx.convert_to_numpy(edge_index).tolist() == [[0, 1], [1, 0]]
    assert tlx.convert_to_numpy(edge_attr).tolist() == [[1, 2], [3, 4]]

# test_remove_self_loops()
#
# def test_segregate_self_loops():
#     edge_index = torch.tensor([[0, 0, 1], [0, 1, 0]])
#
#     out = segregate_self_loops(edge_index)
#     assert out[0].tolist() == [[0, 1], [1, 0]]
#     assert out[1] is None
#     assert out[2].tolist() == [[0], [0]]
#     assert out[3] is None
#
#     edge_attr = torch.tensor([1, 2, 3])
#     out = segregate_self_loops(edge_index, edge_attr)
#     assert out[0].tolist() == [[0, 1], [1, 0]]
#     assert out[1].tolist() == [2, 3]
#     assert out[2].tolist() == [[0], [0]]
#     assert out[3].tolist() == [1]


def test_add_self_loops():
    edge_index = tlx.convert_to_tensor([[0, 1, 0], [1, 0, 0]])
    edge_weight = tlx.convert_to_tensor([0.5, 0.5, 0.5])
    edge_attr = tlx.ops.constant(value=0.5, shape=(3, 3))

    expected = [[0, 1, 0, 0, 1], [1, 0, 0, 0, 1]]
    assert tlx.convert_to_numpy(add_self_loops(edge_index)[0]).tolist() == expected

    out = add_self_loops(edge_index, edge_weight)
    assert tlx.convert_to_numpy(out[0]).tolist() == expected
    assert tlx.convert_to_numpy(out[1]).tolist() == [0.5, 0.5, 0.5, 1., 1.]

    out = add_self_loops(edge_index, edge_weight, fill_value=2.)
    assert tlx.convert_to_numpy(out[0]).tolist() == expected
    assert tlx.convert_to_numpy(out[1]).tolist() == [0.5, 0.5, 0.5, 2., 2.]

    out = add_self_loops(edge_index, edge_weight, fill_value=tlx.convert_to_tensor([2.]))
    assert tlx.convert_to_numpy(out[0]).tolist() == expected
    assert tlx.convert_to_numpy(out[1]).tolist() == [0.5, 0.5, 0.5, 2., 2.]
    #
    # out = add_self_loops(edge_index, edge_weight, fill_value='add')
    # assert out[0].tolist() == expected
    # assert out[1].tolist() == [0.5, 0.5, 0.5, 1, 0.5]

    out = add_self_loops(edge_index, edge_attr)
    assert tlx.convert_to_numpy(out[0]).tolist() == expected
    assert tlx.convert_to_numpy(out[1]).tolist() == [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
                               [1., 1., 1.], [1., 1., 1.]]

    out = add_self_loops(edge_index, edge_attr,
                         fill_value=tlx.convert_to_tensor([0., 1., 0.]))
    assert tlx.convert_to_numpy(out[0]).tolist() == expected
    assert tlx.convert_to_numpy(out[1]).tolist() == [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
                               [0., 1., 0.], [0., 1., 0.]]

    # out = add_self_loops(edge_index, edge_attr, fill_value='add')
    # assert out[0].tolist() == expected
    # assert out[1].tolist() == [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
    #                            [0., 1., 1.], [1., 0., 0.]]

    edge_index = tlx.convert_to_tensor([[], []])
    edge_weight = tlx.convert_to_tensor([], dtype=tlx.float32)
    out = add_self_loops(edge_index, edge_weight, num_nodes=1)
    assert tlx.convert_to_numpy(out[0]).tolist() == [[0], [0]]
    assert tlx.convert_to_numpy(out[1]).tolist() == [1.]


#
# def test_add_remaining_self_loops():
#     edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
#     edge_weight = torch.tensor([0.5, 0.5, 0.5])
#
#     edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight)
#     assert edge_index.tolist() == [[0, 1, 0, 1], [1, 0, 0, 1]]
#     assert edge_weight.tolist() == [0.5, 0.5, 0.5, 1]
#
#
# def test_add_remaining_self_loops_without_initial_loops():
#     edge_index = torch.tensor([[0, 1], [1, 0]])
#     edge_weight = torch.tensor([0.5, 0.5])
#
#     edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight)
#     assert edge_index.tolist() == [[0, 1, 0, 1], [1, 0, 0, 1]]
#     assert edge_weight.tolist() == [0.5, 0.5, 1, 1]
#
#
# def test_get_self_loop_attr():
#     edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
#     edge_weight = torch.tensor([0.2, 0.3, 0.5])
#
#     full_loop_weight = get_self_loop_attr(edge_index, edge_weight)
#     assert full_loop_weight.tolist() == [0.5, 0.0]
#
#     full_loop_weight = get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
#     assert full_loop_weight.tolist() == [0.5, 0.0, 0.0, 0.0]
#
#     full_loop_weight = get_self_loop_attr(edge_index)
#     assert full_loop_weight.tolist() == [1.0, 0.0]
#
#     edge_attr = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 1.0]])
#     full_loop_attr = get_self_loop_attr(edge_index, edge_attr)
#     assert full_loop_attr.tolist() == [[0.5, 1.0], [0.0, 0.0]]