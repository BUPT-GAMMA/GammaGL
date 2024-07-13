from typing import Tuple, Dict, List
import numpy as np
import networkx as nx
from gammagl.data import Graph
from gammagl.utils.convert import to_networkx
from gammagl.utils.shortest_path import floyd_warshall_source_to_all,all_pairs_shortest_path,shortest_path_distance,batched_shortest_path_distance


def test_shortest_path():
    return
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
    num_nodes = 4
    G = nx.Graph()
    G.add_edges_from(edges)
    edge_index = np.array(edges).T
    data = Graph(x=None,edge_index=edge_index)
    G_nx = G
    source = 0
    node_paths, edge_paths = floyd_warshall_source_to_all(G_nx, source)
    print(node_paths)
    print(edge_paths)

    assert node_paths[3] == [0, 1, 3], f"Expected [0, 1, 3], but got {node_paths[3]}"
    assert edge_paths[3] == [0, 3], f"Expected [0, 3], but got {edge_paths[3]}"
    node_paths, edge_paths = all_pairs_shortest_path(G_nx)
    assert node_paths[0][3] == [0, 1, 3], f"Expected [0, 1, 3], but got {node_paths[0][3]}"
    assert edge_paths[0][3] == [0, 3], f"Expected [0, 3], but got {edge_paths[0][3]}"
    node_paths, edge_paths = shortest_path_distance(data)
    assert node_paths[0][3] == [0, 1, 3], f"Expected [0, 1, 3], but got {node_paths[0][3]}"
    assert edge_paths[0][3] == [0, 3], f"Expected [0, 3], but got {edge_paths[0][3]}"
    batch_data = [data, data]  
    node_paths, edge_paths = batched_shortest_path_distance(batch_data)
    assert node_paths[0][3] == [0, 1, 3], f"Expected [0, 1, 3], but got {node_paths[0][3]}"
    assert edge_paths[0][3] == [0, 3], f"Expected [0, 3], but got {edge_paths[0][3]}"
