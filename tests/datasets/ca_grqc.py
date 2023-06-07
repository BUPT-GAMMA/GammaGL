from gammagl.datasets import CA_GrQc


def test_cagrqc():
    datasets = CA_GrQc("ca_grqc", 50)
    assert datasets.n_node == len(datasets.graph)
    assert datasets.n_node == 5242
    assert datasets.node_embed_init_d.shape == (5242, 50)
    assert datasets.node_embed_init_g.shape == (5242, 50)


