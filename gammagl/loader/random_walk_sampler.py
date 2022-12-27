import tensorlayerx as tlx
import numpy as np
from collections import defaultdict
from gammagl.utils.num_nodes import maybe_num_nodes


class RandomWalk:
    def __init__(self, model):
        self.model = model

    def __call__(self, edge_index, num_walks, walk_length, edge_weight=None, p=1.0, q=1.0, num_nodes=None):

        if edge_weight == None:
            edge_weight = tlx.ops.ones(shape=(edge_index.shape[1],), dtype=tlx.float32)
        if num_nodes == None:
            num_nodes = maybe_num_nodes(edge_index)

        src, dst = edge_index[0], edge_index[1]
        src = tlx.convert_to_numpy(src)
        dst = tlx.convert_to_numpy(dst)

        # get source node neighbors.
        node_neighbor = {}
        nodes_weight = {}
        index = 0
        for src_node in src:
            if src_node not in node_neighbor.keys():
                node_neighbor[src_node] = list()

            node_neighbor[src_node].append(dst[index])
            nodes_weight[(src_node, dst[index])] = edge_weight[index]
            index += 1

        walks = list()
        if self.model == "node2vec":
            probs = compute_probabilities(node_neighbor, edge_index, nodes_weight, p, q, num_nodes)
            walks = node2vec_generate_random_walks(node_neighbor, probs, edge_index, num_walks, walk_length)

        elif self.model == "deepwalk":
            walks = deepwalk_generate_random_walks(node_neighbor, edge_index, num_walks, walk_length)

        return walks


def compute_probabilities(neighbor, edge_index, nodes_weight, p, q, num_nodes):
    probs = defaultdict(dict)
    for node in range(num_nodes):
        probs[node]['probabilities'] = dict()

    src = edge_index[0]
    src = tlx.convert_to_numpy(src)
    node = set(src)

    for source_node in node:
        for current_node in neighbor[source_node]:
            probs_ = list()
            for destination in neighbor[current_node]:
                weight = tlx.convert_to_numpy(nodes_weight[(current_node, destination)])
                if source_node == destination:
                    prob_ = (1 / p) * weight
                elif destination in neighbor[source_node]:
                    prob_ = 1 * weight
                else:
                    prob_ = (1 / q) * weight

                probs_.append(prob_)

            probs[source_node]['probabilities'][current_node] = probs_ / np.sum(probs_)

    return probs


def node2vec_generate_random_walks(neighbor, probs, edge_index, num_walks, walk_length):
    src = edge_index[0]
    src = tlx.convert_to_numpy(src)
    node = set(src)

    walks = list()
    for start_node in node:
        for i in range(num_walks):

            walk = [int(start_node)]
            walk_options = neighbor[walk[-1]]
            if len(walk_options) == 0:
                break
            first_step = np.random.choice(walk_options)
            walk.append(int(first_step))

            for k in range(walk_length - 2):
                walk_options = neighbor[walk[-1]]
                if len(walk_options) == 0:
                    break
                probabilities = probs[walk[-2]]['probabilities'][walk[-1]]
                if tlx.BACKEND == 'paddle':
                    probabilities = np.concatenate(probabilities, axis=0)
                next_step = np.random.choice(walk_options, p=probabilities)
                walk.append(int(next_step))

            walks.append(walk)

    np.random.shuffle(walks)

    return walks


def deepwalk_generate_random_walks(neighbor, edge_index, num_walks, walk_length):
    src = edge_index[0]
    src = tlx.convert_to_numpy(src)
    node = set(src)

    walks = list()
    for i in range(num_walks):
        for start_node in node:
            walk = [start_node]
            for k in range(walk_length - 1):
                walk_options = neighbor[walk[-1]]
                if len(walk_options) == 0:
                    break
                next_step = np.random.choice(walk_options)
                walk.append(next_step)
            walks.append(walk)

    np.random.shuffle(walks)

    return walks
