from typing import Tuple, Dict, List
import numpy as np
import networkx as nx
from gammagl.data import Graph
from gammagl.utils.convert import to_networkx
from gammagl.utils.shortest_path import floyd_warshall_source_to_all,all_pairs_shortest_path,shortest_path_distance,batched_shortest_path_distance
# 定义辅助函数来创建 GammaGL Graph 对象
# def create_gammagl_graph(edges, num_nodes):
#     edge_index = np.array(edges).T
#     graph = Graph(x=None, edge_index=edge_index)
#     return graph



def test_shortest_path():
    
    # 示例图数据
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
    num_nodes = 4

    # 创建 NetworkX 图
    G = nx.Graph()
    G.add_edges_from(edges)

    # 转换为 GammaGL 图
    edge_index = np.array(edges).T
    data = Graph(x=None,edge_index=edge_index)
    # 验证 floyd_warshall_source_to_all 函数
    
    G_nx =G
    source = 0
    node_paths, edge_paths = floyd_warshall_source_to_all(G_nx, source)
    assert node_paths[3] == [0, 1, 3], f"Expected [0, 1, 3], but got {node_paths[3]}"
    assert edge_paths[3] == [0, 3], f"Expected [0, 3], but got {edge_paths[3]}"

    # 验证 all_pairs_shortest_path 函数
    node_paths, edge_paths = all_pairs_shortest_path(G_nx)
    assert node_paths[0][3] == [0, 1, 3], f"Expected [0, 1, 3], but got {node_paths[0][3]}"
    assert edge_paths[0][3] == [0, 3], f"Expected [0, 3], but got {edge_paths[0][3]}"

    # 验证 shortest_path_distance 函数
    node_paths, edge_paths = shortest_path_distance(data)
    assert node_paths[0][3] == [0, 1, 3], f"Expected [0, 1, 3], but got {node_paths[0][3]}"
    assert edge_paths[0][3] == [0, 3], f"Expected [0, 3], but got {edge_paths[0][3]}"

    # 验证 batched_shortest_path_distance 函数
    batch_data = [data, data]  # 示例批次数据，包含两个相同的图
    node_paths, edge_paths = batched_shortest_path_distance(batch_data)
    assert node_paths[0][3] == [0, 1, 3], f"Expected [0, 1, 3], but got {node_paths[0][3]}"
    assert edge_paths[0][3] == [0, 3], f"Expected [0, 3], but got {edge_paths[0][3]}"