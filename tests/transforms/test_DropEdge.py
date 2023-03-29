import tensorlayerx as tlx
import numpy as np

from gammagl.data import HeteroGraph
import gammagl.transforms as T
def test_DropEdge():
    p=0.3
    transform=T.DropEdge(p)
    g = HeteroGraph()
    num_papers = 5
    num_paper_features = 6
    num_authors = 7
    num_authors_features = 8
    paper = np.random.uniform(low = 0, high = 1, size = (num_papers, num_paper_features))
    author = np.random.uniform(low = 0, high = 1, size = (num_authors, num_authors_features))
    g['paper'].x = tlx.convert_to_tensor(paper)
    g['author'].x = tlx.convert_to_tensor(author)
    g['author', 'writes', 'paper'].edge_index = tlx.convert_to_tensor([[0, 0, 0], [1, 2, 3]])  # [2, num_edges]
    g['author', 'writes', 'paper'].edge_attr = tlx.convert_to_tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    g['author', 'writes', 'author'].edge_index = tlx.convert_to_tensor(
        [[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6],
         [1, 2, 3, 4, 5, 6, 0, 2, 3, 4, 5, 6, 0, 1, 3, 4, 5, 6, 0, 1, 2, 4, 5, 6, 0, 1, 2, 3]])
    g_edge_num = g.num_edges
    new_g = transform(g)
    new_g_edge_num = new_g.num_edges
    drop_rate=1- new_g_edge_num/g_edge_num
    assert new_g.node_types == g.node_types
    assert new_g.edge_types == g.edge_types
    assert new_g.metadata() == g.metadata()
    # [TODO] Test below value range
    # assert abs(drop_rate-p)<=0.15
