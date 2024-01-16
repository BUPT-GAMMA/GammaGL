
from __future__ import annotations

from typing import Tuple, Dict, List

import networkx as nx
from gammagl.data import Graph
from gammagl.utils.convert import to_networkx


def floyd_warshall_source_to_all(G, source, cutoff=None):
    """
    使用Floyd-Warshall算法计算从源节点到图中所有其他节点的最短路径。

    参数:
    - G: NetworkX图
    - source: 计算最短路径的源节点
    - cutoff: 最短路径的可选截断距离

    返回:
    - node_paths: 字典，将节点映射到表示最短路径的节点列表
    - edge_paths: 字典，将节点映射到表示最短路径的边索引列表
    """
    if source not in G:
        raise nx.NodeNotFound("源节点 {} 不在图中".format(source))

    edges = {edge: i for i, edge in enumerate(G.edges())}

    level = 0  # 当前层级
    nextlevel = {source: 1}  # 下一层级要检查的节点列表
    node_paths = {source: [source]}  # 路径字典（从源到键的路径）
    edge_paths = {source: []}

    while nextlevel:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in G[v]:
                if w not in node_paths:
                    node_paths[w] = node_paths[v] + [w]
                    edge_paths[w] = edge_paths[v] + [edges[tuple(node_paths[w][-2:])]]
                    nextlevel[w] = 1

        level = level + 1

        if cutoff is not None and cutoff <= level:
            break

    return node_paths, edge_paths


def all_pairs_shortest_path(G) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    使用Floyd-Warshall算法计算图中所有节点的所有对最短路径。

    参数:
    - G: NetworkX图

    返回:
    - node_paths: 字典，将节点映射到包含最短路径的字典
    - edge_paths: 字典，将节点映射到包含最短路径的边索引的字典
    """

    # 为每个节点计算最短路径
    paths = {n: floyd_warshall_source_to_all(G, n) for n in G}
    # 将节点映射到最短路径
    node_paths = {n: paths[n][0] for n in paths}
    # 将节点映射到最短路径的边索引
    edge_paths = {n: paths[n][1] for n in paths}
    return node_paths, edge_paths


def shortest_path_distance(data: Graph) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    为torch_geometric数据批次中的每个图计算所有对最短路径。

    参数:
    - data: 包含图的torch_geometric数据

    返回:
    - node_paths: 字典，将节点映射到包含最短路径的字典
    - edge_paths: 字典，将节点映射到包含最短路径的边索引的字典
    """
    G = to_networkx(data)
    node_paths, edge_paths = all_pairs_shortest_path(G)
    return node_paths, edge_paths


def batched_shortest_path_distance(data) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    为torch_geometric数据批次中的每个图计算所有对最短路径。

    参数:
    - data: 包含一批图的torch_geometric数据

    返回:
    - node_paths: 字典，将节点映射到包含最短路径的字典
    - edge_paths: 字典，将节点映射到包含最短路径的边索引的字典
    """
    graphs = [to_networkx(sub_data) for sub_data in data.to_data_list()]
    relabeled_graphs = []
    shift = 0
    for i in range(len(graphs)):
        num_nodes = graphs[i].number_of_nodes()
        relabeled_graphs.append(nx.relabel_nodes(graphs[i], {i: i + shift for i in range(num_nodes)}))
        shift += num_nodes

    paths = [all_pairs_shortest_path(G) for G in relabeled_graphs]
    node_paths = {}
    edge_paths = {}

    for path in paths:
        for k, v in path[0].items():
            node_paths[k] = v
        for k, v in path[1].items():
            edge_paths[k] = v

    return node_paths, edge_paths
