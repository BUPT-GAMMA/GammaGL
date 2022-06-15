import tensorlayerx as tlx
from gammagl.loader import DataLoader

from gammagl.data import Graph


def test_dataloader():
    x = tlx.convert_to_tensor([[1], [1], [1]])
    edge_index = tlx.convert_to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=tlx.int64)
    face = tlx.convert_to_tensor([[0], [1], [2]], dtype=tlx.int64)
    y = 2.
    z = 0.
    name = 'graph'

    data = Graph(x=x, edge_index=edge_index, y=y, z=z, name=name)
    assert str(data) == ("Graph(edge_index=[2, 4], x=[3, 1], y=2.0, z=0.0, name='graph')")
    data.face = face

    loader = DataLoader([data, data, data, data], batch_size=2, shuffle=False,
                        num_workers=0)
    assert len(loader) == 2
    for batch in loader:
        assert len(batch) == 8
        assert tlx.convert_to_numpy(batch.batch).tolist() == [0, 0, 0, 1, 1, 1]
        assert tlx.convert_to_numpy(batch.ptr).tolist() == [0, 3, 6]
        assert tlx.convert_to_numpy(batch.x).tolist() == [[1], [1], [1], [1], [1], [1]]
        assert tlx.convert_to_numpy(batch.edge_index).tolist() == [[0, 1, 1, 2, 3, 4, 4, 5],
                                             [1, 0, 2, 1, 4, 3, 5, 4]]
        assert tlx.convert_to_numpy(batch.y).tolist() == [2.0, 2.0]
        assert tlx.convert_to_numpy(batch.z).tolist() == [0.0, 0.0]
        assert batch.name == ['graph', 'graph']
        assert tlx.convert_to_numpy(batch.face).tolist() == [[0, 3], [1, 4], [2, 5]]

        for store in batch.stores:
            assert id(batch) == id(store._parent())

    loader = DataLoader([data, data, data, data], batch_size=2, shuffle=False,
                        follow_batch=['edge_index'], num_workers=0)
    assert len(loader) == 2

    for batch in loader:
        assert len(batch) == 10
        assert tlx.convert_to_numpy(batch.edge_index_batch).tolist() == [0, 0, 0, 0, 1, 1, 1, 1]

test_dataloader()
