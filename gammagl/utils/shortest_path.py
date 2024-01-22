from typing import Tuple, Dict, List

import networkx as nx
from gammagl.data import Graph
from gammagl.utils.convert import to_networkx


def floyd_warshall_source_to_all(G, source, cutoff=None):
    
    r"""The Floyd-Warshall algorithm is used to calculate the shortest path 
    from the source node to all other nodes in the graph.

    Parameters
    ----------
    G: tensor
        The NetworkX Graph.
    source: tensor
        The source node where the shortest path is calculated.
    cutoff: tensor
        The optional truncation distance of the shortest path.
    
    Returns
    -------
    node_paths: Dict
        The dictionary that maps nodes to a list of nodes that represent the shortest paths.
    edge_paths: Dict
        The dictionary that maps nodes to a list of side indexes representing the shortest path.

    """

    if source not in G:
        raise nx.NodeNotFound("The source node {} is not in the Graph".format(source))

    edges = {edge: i for i, edge in enumerate(G.edges())}

    level = 0
    nextlevel = {source: 1}
    node_paths = {source: [source]}
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
    paths = {n: floyd_warshall_source_to_all(G, n) for n in G}
    node_paths = {n: paths[n][0] for n in paths}
    edge_paths = {n: paths[n][1] for n in paths}
    return node_paths, edge_paths


def shortest_path_distance(data: Graph) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    G = to_networkx(data)
    node_paths, edge_paths = all_pairs_shortest_path(G)
    return node_paths, edge_paths


def batched_shortest_path_distance(data):
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
