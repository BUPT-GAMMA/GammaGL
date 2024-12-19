import numpy as np
import tensorlayerx as tlx
from gammagl.data import Graph


def test_data():
	# torch_geometric.set_debug(True)
	
	x = tlx.convert_to_tensor([[1, 3], [2, 4], [1, 2]], dtype=tlx.float32)
	edge_index = tlx.convert_to_tensor([[0, 0, 1, 1, 2], [1, 1, 0, 2, 1]], dtype=tlx.int64)
	data = Graph(x=x, edge_index=edge_index)

	N = data.num_nodes
	assert N == 3

	assert tlx.convert_to_numpy(data.out_degree).tolist() == [2, 2, 1]
	assert tlx.convert_to_numpy(data.in_degree).tolist() == [1, 3, 1]
	
	assert tlx.convert_to_numpy(data.x).tolist() == tlx.convert_to_numpy(x).tolist()
	assert tlx.convert_to_numpy(data['x']).tolist() == tlx.convert_to_numpy(x).tolist()

	assert sorted(data.keys) == ['edge_index', 'x']
	assert len(data) == 2
	assert 'x' in data and 'edge_index' in data and 'pos' not in data

	D = data.to_dict()
	assert len(D) == 2
	assert 'x' in D and 'edge_index' in D

	D = data.to_namedtuple()
	assert len(D) == 2
	assert D.x is not None and D.edge_index is not None

	assert data.__cat_dim__('x', data.x) == 0
	assert data.__cat_dim__('edge_index', data.edge_index) == -1
	assert data.__inc__('x', data.x) == 0
	assert data.__inc__('edge_index', data.edge_index) == data.num_nodes

	assert not data.is_coalesced()
	data = data.coalesce()
	assert data.is_coalesced()

	# clone = data.clone()
	# assert clone != data
	# assert len(clone) == len(data)
	# assert clone.x.data_ptr() != data.x.data_ptr()
	# assert clone.x.tolist() == data.x.tolist()
	# assert clone.edge_index.data_ptr() != data.edge_index.data_ptr()
	# assert clone.edge_index.tolist() == data.edge_index.tolist()

	# Test `data.to_heterogenous()`:
	out = data.to_heterogeneous()
	assert np.allclose(tlx.convert_to_numpy(data.x), tlx.convert_to_numpy(out['0'].x))
	assert np.allclose(tlx.convert_to_numpy(data.edge_index), tlx.convert_to_numpy(out['0', '0'].edge_index))

	data.edge_type = tlx.convert_to_tensor([0, 0, 1, 0])
	out = data.to_heterogeneous()
	assert np.allclose(tlx.convert_to_numpy(data.x), tlx.convert_to_numpy(out['0'].x))
	assert [store.num_edges for store in out.edge_stores] == [3, 1]
	data.edge_type = None

	data['x'] = x + 1
	assert tlx.convert_to_numpy(data.x).tolist() == tlx.convert_to_numpy(x + 1).tolist()

	assert str(data) == 'Graph(edge_index=[2, 4], x=[3, 2])'

	dictionary = {'x': data.x, 'edge_index': data.edge_index}
	data = Graph.from_dict(dictionary)
	assert sorted(data.keys) == ['edge_index', 'x']

	assert not data.has_isolated_nodes()
	assert not data.has_self_loops()
	assert data.is_undirected()
	assert not data.is_directed()

	assert data.num_nodes == 3
	assert data.num_edges == 4
	assert data.num_node_features == 2
	assert data.num_features == 2

	data.edge_attr = tlx.zeros((data.num_edges, 2))
	assert data.num_edge_features == 2
	data.edge_attr = None

	data.x = None
	assert data.num_nodes == 3

	data.edge_index = None
	assert data.num_nodes is None
	assert data.num_edges == 0

	data.num_nodes = 4
	assert data.num_nodes == 4

	data = Graph(x=x, attribute=x)
	assert len(data) == 2
	assert tlx.convert_to_numpy(data.x).tolist() == tlx.convert_to_numpy(x).tolist()
	assert tlx.convert_to_numpy(data.attribute).tolist() == tlx.convert_to_numpy(x).tolist()

	data = Graph(title='test')
	assert str(data) == "Graph(title='test')"
	assert data.num_node_features == 0
	assert data.num_edge_features == 0

	key = value = 'test_value'
	data[key] = value
	assert data[key] == value
	del data[value]
	del data[value]  # Deleting unset attributes should work as well.

	assert data.get(key) is None
	assert data.get('title') == 'test'

	# torch_geometric.set_debug(False)
	
#
# def test_data_subgraph():
#     x = torch.arange(5)
#     y = torch.tensor([0.])
#     edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
#                                [1, 0, 2, 1, 3, 2, 4, 3]])
#     edge_weight = torch.arange(edge_index.size(1))
#
#     data = Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight,
#                 num_nodes=5)
#
#     out = data.subgraph(torch.tensor([1, 2, 3]))
#     assert len(out) == 5
#     assert torch.allclose(out.x, torch.arange(1, 4))
#     assert torch.allclose(out.y, y)
#     assert out.edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
#     assert torch.allclose(out.edge_weight, edge_weight[torch.arange(2, 6)])
#     assert out.num_nodes == 3
#
#     out = data.subgraph(torch.tensor([False, False, False, True, True]))
#     assert len(out) == 5
#     assert torch.allclose(out.x, torch.arange(3, 5))
#     assert torch.allclose(out.y, y)
#     assert out.edge_index.tolist() == [[0, 1], [1, 0]]
#     assert torch.allclose(out.edge_weight, edge_weight[torch.arange(6, 8)])
#     assert out.num_nodes == 2
#
#
# def test_copy_data():
#     data = Data(x=torch.randn(20, 5))
#
#     out = copy.copy(data)
#     assert id(data) != id(out)
#     assert id(data._store) != id(out._store)
#     assert len(data.stores) == len(out.stores)
#     for store1, store2 in zip(data.stores, out.stores):
#         assert id(store1) != id(store2)
#         assert id(data) == id(store1._parent())
#         assert id(out) == id(store2._parent())
#     assert data.x.data_ptr() == out.x.data_ptr()
#
#     out = copy.deepcopy(data)
#     assert id(data) != id(out)
#     assert id(data._store) != id(out._store)
#     assert len(data.stores) == len(out.stores)
#     for store1, store2 in zip(data.stores, out.stores):
#         assert id(store1) != id(store2)
#         assert id(data) == id(store1._parent())
#         assert id(out) == id(store2._parent())
#     assert data.x.data_ptr() != out.x.data_ptr()
#     assert data.x.tolist() == out.x.tolist()
#
#
# def test_debug_data():
#     torch_geometric.set_debug(True)
#
#     Data()
#     Data(edge_index=torch.tensor([[0, 1], [1, 0]])).num_nodes
#     Data(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=10)
#     Data(face=torch.zeros((3, 0), dtype=torch.long), num_nodes=10)
#     Data(edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.randn(2))
#     Data(face=torch.tensor([[0], [1], [2]])).num_nodes
#     Data(x=torch.torch.randn(5, 3), num_nodes=5)
#     Data(pos=torch.torch.randn(5, 3), num_nodes=5)
#     Data(norm=torch.torch.randn(5, 3), num_nodes=5)
#
#     torch_geometric.set_debug(False)
#
#
# def run(rank, data_list):
#     for data in data_list:
#         assert data.x.is_shared()
#         data.x.add_(1)
#
#
# def test_data_share_memory():
#     data_list = [Data(x=torch.zeros(8)) for _ in range(10)]
#
#     for data in data_list:
#         assert not data.x.is_shared()
#
#     mp.spawn(run, args=(data_list, ), nprocs=4, join=True)
#
#     for data in data_list:
#         assert data.x.is_shared()
#         assert torch.allclose(data.x, torch.full((8, ), 4.))
