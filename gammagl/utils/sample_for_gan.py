import numpy as np


def sample(all_score, root, tree, sample_num, for_d):
    """ sample nodes from BFS-tree

    Args:
        root: int, root node
        tree: dict, BFS-tree
        sample_num: the number of required samples
        for_d: bool, whether the samples are used for the G or the D

    Returns:
        samples: list, the indices of the sampled nodes
        paths: list, paths from the root to the sampled nodes

    """
    samples = []
    paths = []
    n = 0

    while len(samples) < sample_num:
        current_node = root
        previous_node = -1
        paths.append([])
        is_root = True
        paths[n].append(current_node)
        while True:
            node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
            is_root = False
            if len(node_neighbor) == 0:  # the tree only has a root
                return None, None
            if for_d:  # skip 1-hop nodes (positive samples)
                if node_neighbor == [root]:
                    # in current version, None is returned for simplicity
                    return None, None
                if root in node_neighbor:
                    node_neighbor.remove(root)
            relevance_probability = all_score[current_node, node_neighbor]
            e_x = np.exp(relevance_probability - np.max(relevance_probability))
            relevance_probability = e_x / e_x.sum()
            next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]  # select next node
            paths[n].append(next_node)
            if next_node == previous_node:  # terminating condition
                samples.append(current_node)
                break
            previous_node = current_node
            current_node = next_node
        n = n + 1
    return samples, paths
