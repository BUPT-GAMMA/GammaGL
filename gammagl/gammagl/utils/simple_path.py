def find_all_simple_paths(edge_index, src, dest, max_length):
    r"""
        The :obj:`find_all_simple_paths` function is used to find all simple paths (that is, paths that do not duplicate nodes) from the source node to the destination node in a given graph. The function accepts as parameters the edge index of the graph, the maximum length of the source node, the target node, and the path, and returns a list of all simple paths from the source node to the target node that do not exceed the maximum length.

        Parameters
        ----------
        edge_index: tensor
            A 2-D Tensor of the edge index of a graph with the shape [2, num_edges]. Each column contains two end-point indexes for one side.
        src: tensor
            The index of the source node.
        dest: tensor
            The index of the target node.
        max_length: int
            The maximum length of a path.

        Return
        -------
        list[list[int]]
            A list of all the simple paths from the source node to the destination node.
    """
    num_nodes = max(edge_index[0].max().item(),
                    edge_index[1].max().item(),
                    -edge_index[0].min().item(),
                    -edge_index[1].min().item(),
                    abs(src.item())) + 1
    adj_list = [[] for _ in range(num_nodes)]
    for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adj_list[u].append(v)

    src = src.item()

    paths = []
    visited = set()
    stack = [(src, [src])]

    while stack:
        (node, path) = stack.pop()

        if node == dest:
            paths.append(path)
        elif len(path) < max_length:
            for neighbor in adj_list[node]:
                if neighbor not in path:
                    visited.add((node, neighbor))
                    stack.append((neighbor, path + [neighbor]))
            for neighbor in adj_list[node]:
                if (node, neighbor) in visited:
                    visited.remove((node, neighbor))

    return paths
