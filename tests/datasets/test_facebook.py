from gammagl.datasets import FacebookPagePage

def test_facebook():
    dataset = FacebookPagePage(root='data/facebook')
    g = dataset[0]
    assert len(dataset) == 1
    assert g.num_nodes == 22470
    assert g.num_edges == 342004
    assert g.x.shape[1] == 128
    assert len(set(g.y.numpy())) == 4

