import numpy as np
import tensorlayerx as tlx

from gammagl.data import BatchGraph, Graph, HeteroGraph


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = tlx.ops.convert_to_tensor(np.random.randint(num_src_nodes, size=(num_edges,), dtype=np.int64))
    col = tlx.ops.convert_to_tensor(np.random.randint(num_dst_nodes, size=(num_edges,), dtype=np.int64))
    return tlx.stack([row, col], axis=0)


def test_batch():
    # torch_geometric.set_debug(True)

    x1 = tlx.convert_to_tensor([1, 2, 3], dtype=tlx.float32)
    y1 = 1
    # x1_sp = SparseTensor.from_dense(x1.view(-1, 1))
    e1 = tlx.convert_to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=tlx.int64)
    # adj1 = SparseTensor.from_edge_index(e1)
    s1 = '1'
    array1 = ['1', '2']
    x2 = tlx.convert_to_tensor([1, 2], dtype=tlx.float32)
    y2 = 2
    # x2_sp = SparseTensor.from_dense(x2.view(-1, 1))
    e2 = tlx.convert_to_tensor([[0, 1], [1, 0]], dtype=tlx.int64)
    # adj2 = SparseTensor.from_edge_index(e2)
    s2 = '2'
    array2 = ['3', '4', '5']
    x3 = tlx.convert_to_tensor([1, 2, 3, 4], dtype=tlx.float32)
    y3 = 3
    # x3_sp = SparseTensor.from_dense(x3.view(-1, 1))
    e3 = tlx.convert_to_tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=tlx.int64)
    # adj3 = SparseTensor.from_edge_index(e3)
    s3 = '3'
    array3 = ['6', '7', '8', '9']

    # data1 = Data(x=x1, y=y1, x_sp=x1_sp, edge_index=e1, adj=adj1, s=s1,
    #              array=array1, num_nodes=3)
    # data2 = Data(x=x2, y=y2, x_sp=x2_sp, edge_index=e2, adj=adj2, s=s2,
    #              array=array2, num_nodes=2)
    # data3 = Data(x=x3, y=y3, x_sp=x3_sp, edge_index=e3, adj=adj3, s=s3,
    #              array=array3, num_nodes=4)
    data1 = Graph(x=x1, y=y1, edge_index=e1, s=s1, array=array1, num_nodes=3)
    data2 = Graph(x=x2, y=y2, edge_index=e2, s=s2, array=array2, num_nodes=2)
    data3 = Graph(x=x3, y=y3, edge_index=e3, s=s3, array=array3, num_nodes=4)

    batch = BatchGraph.from_data_list([data1])
    assert str(batch) == ('GraphBatchGraph(edge_index=[2, 4], x=[3], y=[1], s=[1], array=[1], num_nodes=3, batch=[3], ptr=[2])')
    assert batch.num_graphs == 1
    assert len(batch) == 8
    assert tlx.convert_to_numpy(batch.x).tolist() == [1, 2, 3]
    assert tlx.convert_to_numpy(batch.y).tolist() == [1]
    # assert batch.x_sp.to_dense().view(-1).tolist() == batch.x.tolist()
    assert tlx.convert_to_numpy(batch.edge_index).tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    # edge_index = torch.stack(batch.adj.coo()[:2], dim=0)
    # assert edge_index.tolist() == batch.edge_index.tolist()
    assert batch.s == ['1']
    assert batch.array == [['1', '2']]
    assert batch.num_nodes == 3
    assert tlx.convert_to_numpy(batch.batch).tolist() == [0, 0, 0]
    assert tlx.convert_to_numpy(batch.ptr).tolist() == [0, 3]

    batch = BatchGraph.from_data_list([data1, data2, data3], follow_batch=['s'])

    assert str(batch) == ('GraphBatchGraph(edge_index=[2, 12], x=[9], y=[3], s=[3], s_batch=[3], s_ptr=[4], array=[3], num_nodes=9, batch=[9], ptr=[4])')
    assert batch.num_graphs == 3
    assert len(batch) == 10
    assert tlx.convert_to_numpy(batch.x).tolist() == [1, 2, 3, 1, 2, 1, 2, 3, 4]
    assert tlx.convert_to_numpy(batch.y).tolist() == [1, 2, 3]
    # assert batch.x_sp.to_dense().view(-1).tolist() == batch.x.tolist()
    assert tlx.convert_to_numpy(batch.edge_index).tolist() == [[0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8],
                                         [1, 0, 2, 1, 4, 3, 6, 5, 7, 6, 8, 7]]
    # edge_index = torch.stack(batch.adj.coo()[:2], dim=0)
    # assert edge_index.tolist() == batch.edge_index.tolist()
    assert batch.s == ['1', '2', '3']
    assert tlx.convert_to_numpy(batch.s_batch).tolist() == [0, 1, 2]
    assert tlx.convert_to_numpy(batch.s_ptr).tolist() == [0, 1, 2, 3]
    assert batch.array == [['1', '2'], ['3', '4', '5'], ['6', '7', '8', '9']]
    assert batch.num_nodes == 9
    assert tlx.convert_to_numpy(batch.batch).tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    assert tlx.convert_to_numpy(batch.ptr).tolist() == [0, 3, 5, 9]

    data = batch[0]
    assert str(data) == ("Graph(edge_index=[2, 4], x=[3], y=[1], s='1', array=[2], num_nodes=3)")
    data = batch[1]
    assert str(data) == ("Graph(edge_index=[2, 2], x=[2], y=[1], s='2', "
                         "array=[3], num_nodes=2)")

    data = batch[2]
    assert str(data) == ("Graph(edge_index=[2, 6], x=[4], y=[1], s='3', "
                         "array=[4], num_nodes=4)")

    assert len(batch.index_select([1, 0])) == 2
    assert len(batch.index_select(tlx.convert_to_tensor([1, 0], dtype=tlx.int64))) == 2
    assert len(batch.index_select(tlx.convert_to_tensor([True, False]))) == 1
    assert len(batch.index_select(np.array([1, 0], dtype=np.int64))) == 2
    assert len(batch.index_select(np.array([True, False]))) == 1
    assert len(batch[:2]) == 2

    data_list = batch.to_data_list()
    assert len(data_list) == 3

    assert len(data_list[0]) == 6   # 8-2
    assert tlx.convert_to_numpy(data_list[0].x).tolist() == [1, 2, 3]
    assert tlx.convert_to_numpy(data_list[0].y).tolist() == [1]
    # assert data_list[0].x_sp.to_dense().view(-1).tolist() == [1, 2, 3]
    assert tlx.convert_to_numpy(data_list[0].edge_index).tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    # edge_index = torch.stack(data_list[0].adj.coo()[:2], dim=0)
    # assert edge_index.tolist() == data_list[0].edge_index.tolist()
    assert data_list[0].s == '1'
    assert data_list[0].array == ['1', '2']
    assert data_list[0].num_nodes == 3

    assert len(data_list[1]) == 6   # 8-2
    assert tlx.convert_to_numpy(data_list[1].x).tolist() == [1, 2]
    assert tlx.convert_to_numpy(data_list[1].y).tolist() == [2]
    # assert data_list[1].x_sp.to_dense().view(-1).tolist() == [1, 2]
    assert tlx.convert_to_numpy(data_list[1].edge_index).tolist() == [[0, 1], [1, 0]]
    # edge_index = torch.stack(data_list[1].adj.coo()[:2], dim=0)
    # assert edge_index.tolist() == data_list[1].edge_index.tolist()
    assert data_list[1].s == '2'
    assert data_list[1].array == ['3', '4', '5']
    assert data_list[1].num_nodes == 2

    assert len(data_list[2]) == 6   # 8-2
    assert tlx.convert_to_numpy(data_list[2].x).tolist() == [1, 2, 3, 4]
    assert tlx.convert_to_numpy(data_list[2].y).tolist() == [3]
    # assert data_list[2].x_sp.to_dense().view(-1).tolist() == [1, 2, 3, 4]
    assert tlx.convert_to_numpy(data_list[2].edge_index).tolist() == [[0, 1, 1, 2, 2, 3],
                                                [1, 0, 2, 1, 3, 2]]
    # edge_index = torch.stack(data_list[2].adj.coo()[:2], dim=0)
    # assert edge_index.tolist() == data_list[2].edge_index.tolist()
    assert data_list[2].s == '3'
    assert data_list[2].array == ['6', '7', '8', '9']
    assert data_list[2].num_nodes == 4

    # torch_geometric.set_debug(True)
def test_batching_with_new_dimension():
    # torch_geometric.set_debug(True)

    class MyData(Graph):
        def __cat_dim__(self, key, value, *args, **kwargs):
            if key == 'foo':
                return None
            else:
                return super().__cat_dim__(key, value, *args, **kwargs)

    x1 = tlx.convert_to_tensor([1, 2, 3], dtype=tlx.float32)
    foo1 = tlx.convert_to_tensor(np.random.randn(4), dtype=tlx.float32)
    y1 = tlx.convert_to_tensor([1], dtype=tlx.float32)

    x2 = tlx.convert_to_tensor([1, 2], dtype=tlx.float32)
    foo2 = tlx.convert_to_tensor(np.random.randn(4), dtype=tlx.float32)
    y2 = tlx.convert_to_tensor([2], dtype=tlx.float32)

    batch = BatchGraph.from_data_list(
        [MyData(x=x1, foo=foo1, y=y1),
         MyData(x=x2, foo=foo2, y=y2)])

    assert str(batch) == ('MyDataBatchGraph(x=[5], y=[2], foo=[2, 4], batch=[5], ptr=[3])')
    assert len(batch) == 5
    assert tlx.convert_to_numpy(batch.x).tolist() == [1, 2, 3, 1, 2]
    assert tlx.get_tensor_shape(batch.foo) == [2, 4]
    assert tlx.convert_to_numpy(batch.foo[0]).tolist() == tlx.convert_to_numpy(foo1).tolist()
    assert tlx.convert_to_numpy(batch.foo[1]).tolist() == tlx.convert_to_numpy(foo2).tolist()
    assert tlx.convert_to_numpy(batch.y).tolist() == [1, 2]
    assert tlx.convert_to_numpy(batch.batch).tolist() == [0, 0, 0, 1, 1]
    assert tlx.convert_to_numpy(batch.ptr).tolist() == [0, 3, 5]
    assert batch.num_graphs == 2

    data = batch[0]
    assert str(data) == ('MyData(x=[3], y=[1], foo=[4])')
    data = batch[1]
    assert str(data) == ('MyData(x=[2], y=[1], foo=[4])')

    # torch_geometric.set_debug(True)



def test_recursive_batch():
    data1 = Graph(
        x={
            '1': tlx.convert_to_tensor(np.random.randn(10, 32), dtype=tlx.float32),
            '2': tlx.convert_to_tensor(np.random.randn(20, 48), dtype=tlx.float32)
        }, edge_index=[get_edge_index(30, 30, 50),
                       get_edge_index(30, 30, 70)], num_nodes=30)

    data2 = Graph(
        x={
            '1': tlx.convert_to_tensor(np.random.randn(20, 32), dtype=tlx.float32),
            '2': tlx.convert_to_tensor(np.random.randn(40, 48), dtype=tlx.float32)
        }, edge_index=[get_edge_index(60, 60, 80),
                       get_edge_index(60, 60, 90)], num_nodes=60)

    batch = BatchGraph.from_data_list([data1, data2])

    assert len(batch) == 5
    assert batch.num_graphs == 2
    assert batch.num_nodes == 90

    assert np.allclose(tlx.convert_to_numpy(batch.x['1']),
                          tlx.convert_to_numpy(tlx.concat([data1.x['1'], data2.x['1']], axis=0)))
    assert np.allclose(tlx.convert_to_numpy(batch.x['2']),
                          tlx.convert_to_numpy(tlx.concat([data1.x['2'], data2.x['2']], axis=0)))
    assert (tlx.convert_to_numpy(batch.edge_index[0]).tolist() == tlx.convert_to_numpy(tlx.concat(
        [data1.edge_index[0], data2.edge_index[0] + 30], axis=1)).tolist())
    assert (tlx.convert_to_numpy(batch.edge_index[1]).tolist() == tlx.convert_to_numpy(tlx.concat(
        [data1.edge_index[1], data2.edge_index[1] + 30], axis=1)).tolist())
    assert tlx.get_tensor_shape(batch.batch) == [90, ]
    assert tlx.get_tensor_shape(batch.ptr) == [3, ]

    out1 = batch[0]
    assert len(out1) == 3
    assert out1.num_nodes == 30
    assert np.allclose(tlx.convert_to_numpy(out1.x['1']), tlx.convert_to_numpy(data1.x['1']))
    assert np.allclose(tlx.convert_to_numpy(out1.x['2']), tlx.convert_to_numpy(data1.x['2']))
    assert tlx.convert_to_numpy(out1.edge_index[0]).tolist(), tlx.convert_to_numpy(data1.edge_index[0]).tolist()
    assert tlx.convert_to_numpy(out1.edge_index[1]).tolist(), tlx.convert_to_numpy(data1.edge_index[1]).tolist()

    out2 = batch[1]
    assert len(out2) == 3
    assert out2.num_nodes == 60
    assert np.allclose(tlx.convert_to_numpy(out2.x['1']), tlx.convert_to_numpy(data2.x['1']))
    assert np.allclose(tlx.convert_to_numpy(out2.x['2']), tlx.convert_to_numpy(data2.x['2']))
    assert tlx.convert_to_numpy(out2.edge_index[0]).tolist(), tlx.convert_to_numpy(data2.edge_index[0]).tolist()
    assert tlx.convert_to_numpy(out2.edge_index[1]).tolist(), tlx.convert_to_numpy(data2.edge_index[1]).tolist()


def test_batching_of_batches():
    data = Graph(x=tlx.convert_to_tensor(np.random.randn(2, 16)))
    batch = BatchGraph.from_data_list([data, data])

    batch = BatchGraph.from_data_list([batch, batch])
    assert len(batch) == 2
    assert tlx.convert_to_numpy(batch.x[0:2]).tolist() == tlx.convert_to_numpy(data.x).tolist()
    assert tlx.convert_to_numpy(batch.x[2:4]).tolist() == tlx.convert_to_numpy(data.x).tolist()
    assert tlx.convert_to_numpy(batch.x[4:6]).tolist() == tlx.convert_to_numpy(data.x).tolist()
    assert tlx.convert_to_numpy(batch.x[6:8]).tolist() == tlx.convert_to_numpy(data.x).tolist()
    assert tlx.convert_to_numpy(batch.batch).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_hetero_batch():
    e1 = ('p', 'a')
    e2 = ('a', 'p')
    data1 = HeteroGraph()
    data1['p'].x = tlx.convert_to_tensor(np.random.randn(100, 128))
    data1['a'].x = tlx.convert_to_tensor(np.random.randn(200, 128))
    data1[e1].edge_index = get_edge_index(100, 200, 500)
    data1[e1].edge_attr = tlx.convert_to_tensor(np.random.randn(500, 32))
    data1[e2].edge_index = get_edge_index(200, 100, 400)
    data1[e2].edge_attr = tlx.convert_to_tensor(np.random.randn(400, 32))

    data2 = HeteroGraph()
    data2['p'].x = tlx.convert_to_tensor(np.random.randn(50, 128))
    data2['a'].x = tlx.convert_to_tensor(np.random.randn(100, 128))
    data2[e1].edge_index = get_edge_index(50, 100, 300)
    data2[e1].edge_attr = tlx.convert_to_tensor(np.random.randn(300, 32))
    data2[e2].edge_index = get_edge_index(100, 50, 200)
    data2[e2].edge_attr = tlx.convert_to_tensor(np.random.randn(200, 32))

    batch = BatchGraph.from_data_list([data1, data2])

    assert len(batch) == 5
    assert batch.num_graphs == 2
    assert batch.num_nodes == 450

    assert np.allclose(tlx.convert_to_numpy(batch['p'].x[:100]), tlx.convert_to_numpy(data1['p'].x))
    assert np.allclose(tlx.convert_to_numpy(batch['a'].x[:200]), tlx.convert_to_numpy(data1['a'].x))
    assert np.allclose(tlx.convert_to_numpy(batch['p'].x[100:]), tlx.convert_to_numpy(data2['p'].x))
    assert np.allclose(tlx.convert_to_numpy(batch['a'].x[200:]), tlx.convert_to_numpy(data2['a'].x))
    assert (tlx.convert_to_numpy(batch[e1].edge_index).tolist() == tlx.convert_to_numpy(tlx.concat([
        data1[e1].edge_index,
        data2[e1].edge_index + tlx.convert_to_tensor([[100], [200]], dtype=tlx.int64)
    ], 1)).tolist())
    assert np.allclose(
        tlx.convert_to_numpy(batch[e1].edge_attr),
        tlx.convert_to_numpy(tlx.concat([data1[e1].edge_attr, data2[e1].edge_attr], axis=0)))
    assert (tlx.convert_to_numpy(batch[e2].edge_index).tolist() == tlx.convert_to_numpy(tlx.concat([
        data1[e2].edge_index,
        data2[e2].edge_index + tlx.convert_to_tensor([[200], [100]], dtype=tlx.int64)
    ], 1)).tolist())
    assert np.allclose(
        tlx.convert_to_numpy(batch[e2].edge_attr),
        tlx.convert_to_numpy(tlx.concat([data1[e2].edge_attr, data2[e2].edge_attr], axis=0)))
    assert tlx.get_tensor_shape(batch['p'].batch) == [150, ]
    assert tlx.get_tensor_shape(batch['p'].ptr) == [3, ]
    assert tlx.get_tensor_shape(batch['a'].batch) == [300, ]
    assert tlx.get_tensor_shape(batch['a'].ptr) == [3, ]

    out1 = batch[0]
    assert len(out1) == 3
    assert out1.num_nodes == 300
    assert np.allclose(tlx.convert_to_numpy(out1['p'].x), tlx.convert_to_numpy(data1['p'].x))
    assert np.allclose(tlx.convert_to_numpy(out1['a'].x), tlx.convert_to_numpy(data1['a'].x))
    assert tlx.convert_to_numpy(out1[e1].edge_index).tolist() == tlx.convert_to_numpy(data1[e1].edge_index).tolist()
    assert np.allclose(tlx.convert_to_numpy(out1[e1].edge_attr), tlx.convert_to_numpy(data1[e1].edge_attr))
    assert tlx.convert_to_numpy(out1[e2].edge_index).tolist() == tlx.convert_to_numpy(data1[e2].edge_index).tolist()
    assert np.allclose(tlx.convert_to_numpy(out1[e2].edge_attr), tlx.convert_to_numpy(data1[e2].edge_attr))

    out2 = batch[1]
    assert len(out2) == 3
    assert out2.num_nodes == 150
    assert np.allclose(tlx.convert_to_numpy(out2['p'].x), tlx.convert_to_numpy(data2['p'].x))
    assert np.allclose(tlx.convert_to_numpy(out2['a'].x), tlx.convert_to_numpy(data2['a'].x))
    assert tlx.convert_to_numpy(out2[e1].edge_index).tolist() == tlx.convert_to_numpy(data2[e1].edge_index).tolist()
    assert np.allclose(tlx.convert_to_numpy(out2[e1].edge_attr), tlx.convert_to_numpy(data2[e1].edge_attr))
    assert tlx.convert_to_numpy(out2[e2].edge_index).tolist() == tlx.convert_to_numpy(data2[e2].edge_index).tolist()
    assert np.allclose(tlx.convert_to_numpy(out2[e2].edge_attr), tlx.convert_to_numpy(data2[e2].edge_attr))


def test_pair_data_batching():
    class PairData(Graph):
        def __inc__(self, key, value, *args, **kwargs):
            if key == 'edge_index_s':
                return tlx.get_tensor_shape(self.x_s)[0]
            if key == 'edge_index_t':
                return tlx.get_tensor_shape(self.x_t)[0]
            else:
                return super().__inc__(key, value, *args, **kwargs)

    x_s = tlx.convert_to_tensor(np.random.randn(5, 16))
    edge_index_s = tlx.convert_to_tensor([
        [0, 0, 0, 0],
        [1, 2, 3, 4],
    ], dtype=tlx.int64)
    x_t = tlx.convert_to_tensor(np.random.randn(4, 16))
    edge_index_t = tlx.convert_to_tensor([
        [0, 0, 0],
        [1, 2, 3],
    ], dtype=tlx.int64)

    data = PairData(x_s=x_s, edge_index_s=edge_index_s, x_t=x_t,
                    edge_index_t=edge_index_t)
    batch = BatchGraph.from_data_list([data, data])

    assert np.allclose(tlx.convert_to_numpy(batch.x_s), tlx.convert_to_numpy(tlx.concat([x_s, x_s], axis=0)))
    assert tlx.convert_to_numpy(batch.edge_index_s).tolist() == [[0, 0, 0, 0, 5, 5, 5, 5],
                                           [1, 2, 3, 4, 6, 7, 8, 9]]

    assert np.allclose(tlx.convert_to_numpy(batch.x_t), tlx.convert_to_numpy(tlx.concat([x_t, x_t], axis=0)))
    assert tlx.convert_to_numpy(batch.edge_index_t).tolist() == [[0, 0, 0, 4, 4, 4],
                                           [1, 2, 3, 5, 6, 7]]


def test_batch_with_empty_list():
    #  https://github.com/pyg-team/pytorch_geometric/commit/7cff9839a8e8e4182438c7bb8d2427077fbc836c
    x = tlx.convert_to_tensor(np.random.randn(4, 1))
    edge_index = tlx.convert_to_tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=tlx.int64)
    data = Graph(x=x, edge_index=edge_index, nontensor=[])

    batch = BatchGraph.from_data_list([data, data])
    assert batch.nontensor == [[], []]
    assert batch[0].nontensor == []
    assert batch[1].nontensor == []

