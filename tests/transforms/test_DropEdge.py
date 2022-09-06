import tensorlayerx as tlx

from gammagl.data import HeteroGraph
import gammagl.transforms as T
def test_DropEdge():
    transform=T.DropEdge()
    g = HeteroGraph()
    num_papers = 5
    num_paper_features = 6
    num_authors = 7
    num_authors_features = 8
    g['paper'].x = tlx.random_uniform((num_papers, num_paper_features))
    g['author'].x = tlx.random_uniform((num_authors, num_authors_features))
    g['author', 'writes', 'paper'].edge_index = tlx.convert_to_tensor([[0, 0, 0], [1, 2, 3]])  # [2, num_edges]
    g['author', 'writes', 'paper'].edge_attr = tlx.convert_to_tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    g['author', 'writes', 'author'].edge_index = tlx.convert_to_tensor([[1, 2, 3], [2, 3, 1]])
    new_g = transform(g)
    assert new_g.node_types == g.node_types
    assert new_g.edge_types == g.edge_types
    assert new_g.metadata() == g.metadata()


