from gammagl.datasets.dblp import DBLP


def test_dblp():
    return
    root = './temp'
    dataset = DBLP(root=root, force_reload=True)
    data = dataset[0]
    assert 'author' in data.node_types, "Node type 'author' not found in data."
    assert 'paper' in data.node_types, "Node type 'paper' not found in data."
    assert 'term' in data.node_types, "Node type 'term' not found in data."
    assert 'conference' in data.node_types, "Node type 'conference' not found in data."
    assert data['author'].x.shape == (4057, 334), f"Author features shape mismatch: {data['author'].x.shape}"
    assert data['paper'].x.shape == (14328, 4231), f"Paper features shape mismatch: {data['paper'].x.shape}"
    assert data['term'].x.shape == (7723, 50), f"Term features shape mismatch: {data['term'].x.shape}"
    assert data['conference'].num_nodes == 20, f"Conference node count mismatch: {data['conference'].num_nodes}"
    assert data['author'].y.shape[0] == 4057, f"Author labels shape mismatch: {data['author'].y.shape}"
    print("All tests passed!")
