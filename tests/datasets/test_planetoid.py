import tensorlayerx as tlx
from gammagl.loader import DataLoader


def test_citeseer(get_dataset):
    dataset = get_dataset(name='CiteSeer')
    loader = DataLoader(dataset, batch_size=len(dataset))

    assert len(dataset) == 1
    assert dataset.__repr__() == 'CiteSeer()'

    for data in loader:
        assert data.num_graphs == 1
        assert data.num_nodes == 3327
        assert data.num_edges / 2 == 4552

        assert len(data) == 8
        assert list(data.x.shape) == [data.num_nodes, 3703]
        assert list(data.y.shape) == [data.num_nodes]
        assert int(tlx.reduce_max(data.y) + 1) == 6
        assert tlx.reduce_sum(tlx.cast(data.train_mask, dtype=tlx.int64)) == 6 * 20
        assert tlx.reduce_sum(tlx.cast(data.val_mask, dtype=tlx.int64)) == 500
        assert tlx.reduce_sum(tlx.cast(data.test_mask, dtype=tlx.int64)) == 1000
        # [TODO] Mindspore do not support operators &
        mask = tlx.logical_and(tlx.logical_and(data.train_mask, data.val_mask), data.test_mask)
        assert tlx.reduce_sum(tlx.cast(mask, dtype=tlx.int64)) == 0
        # assert tlx.reduce_sum(tlx.cast((data.train_mask & data.val_mask & data.test_mask), dtype=tlx.int64)) == 0
        assert list(data.batch.shape) == [data.num_nodes]
        assert list(data.ptr) == [0, data.num_nodes]

        assert data.has_isolated_nodes()
        assert not data.has_self_loops()
        # [TODO] is_undirected needs tlx.argsort() 
        assert data.is_undirected()


def test_citeseer_with_full_split(get_dataset):
    dataset = get_dataset(name='CiteSeer', split='full')
    data = dataset[0]
    assert tlx.reduce_sum(tlx.cast(data.val_mask, dtype=tlx.int64)) == 500
    assert tlx.reduce_sum(tlx.cast(data.test_mask, dtype=tlx.int64)) == 1000
    assert tlx.reduce_sum(tlx.cast(data.train_mask, dtype=tlx.int64)) == data.num_nodes - 1500
    # [TODO] Mindspore do not support operators &
    mask = tlx.logical_and(tlx.logical_and(data.train_mask, data.val_mask), data.test_mask)
    assert tlx.reduce_sum(tlx.cast(mask, dtype=tlx.int64)) == 0
    # assert tlx.reduce_sum(tlx.cast((data.train_mask & data.val_mask & data.test_mask), dtype=tlx.int64)) == 0


def test_citeseer_with_random_split(get_dataset):
    dataset = get_dataset(name='CiteSeer', split='random',
                          num_train_per_class=11, num_val=29, num_test=41)
    data = dataset[0]
    assert tlx.reduce_sum(tlx.cast(data.train_mask, dtype=tlx.int64)) == dataset.num_classes * 11
    assert tlx.reduce_sum(tlx.cast(data.val_mask, dtype=tlx.int64)) == 29
    assert tlx.reduce_sum(tlx.cast(data.test_mask, dtype=tlx.int64)) == 41
    # [TODO] Mindspore do not support operators &
    mask = tlx.logical_and(tlx.logical_and(data.train_mask, data.val_mask), data.test_mask)
    assert tlx.reduce_sum(tlx.cast(mask, dtype=tlx.int64)) == 0
    # assert tlx.reduce_sum(tlx.cast((data.train_mask & data.val_mask & data.test_mask), dtype=tlx.int64)) == 0
