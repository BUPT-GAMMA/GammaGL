# !/usr/bin/env python
# -*- encoding: utf-8 -*-
import copy
import glob
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ['TL_BACKEND'] = 'paddle'
import sys

import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep
from tensorlayerx.nn import Module
from gammagl.utils import read_edges, read_embeddings, read_edges_from_file

import pickle
import collections
import multiprocessing
import tqdm as tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from functools import partial

import config

sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.


def construct_trees_with_mp(nodes, n_node, trees, graph):
    """use the multiprocessing to speed up trees construction

    Args:
        nodes: the list of nodes in the graph
    """

    cores = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(cores)
    new_nodes = []
    n_node_per_core = n_node // cores
    for i in range(cores):
        if i != cores - 1:
            new_nodes.append(nodes[i * n_node_per_core: (i + 1) * n_node_per_core])
        else:
            new_nodes.append(nodes[i * n_node_per_core:])
    func = partial(construct_trees, graph)
    trees_result = pool.map(func, new_nodes)
    for tree in trees_result:
        trees.update(tree)


def construct_trees(graph, nodes):
    """use BFS algorithm to construct the BFS-trees

    Args:
        nodes: the list of nodes in the graph
    Returns:
        trees: dict, root_node_id -> tree, where tree is a dict: node_id -> list: [father, child_0, child_1, ...]
    """

    trees = {}
    for root in tqdm.tqdm(nodes):
        trees[root] = {}
        trees[root][root] = [root]
        used_nodes = set()
        queue = collections.deque([root])
        while len(queue) > 0:
            cur_node = queue.popleft()
            used_nodes.add(cur_node)
            for sub_node in graph[cur_node]:
                if sub_node not in used_nodes:
                    trees[root][cur_node].append(sub_node)
                    trees[root][sub_node] = [cur_node]
                    queue.append(sub_node)
                    used_nodes.add(sub_node)
    return trees


def calculate_acc(n_node):
    results = []
    if config.app == "link_prediction":
        for i in range(2):
            test_filename = config.test_filename
            test_neg_filename = config.test_neg_filename
            embed_filename = config.emb_filenames[i]
            n_emb = config.n_emb
            emd = read_embeddings(embed_filename, n_node=n_node, n_embed=n_emb)

            test_edges = read_edges_from_file(test_filename)
            test_edges_neg = read_edges_from_file(test_neg_filename)
            test_edges.extend(test_edges_neg)

            # may exists isolated point
            score_res = []
            for j in range(len(test_edges)):
                score_res.append(np.dot(emd[test_edges[j][0]], emd[test_edges[j][1]]))
            test_label = np.array(score_res)
            median = np.median(test_label)
            index_pos = test_label >= median
            index_neg = test_label < median
            test_label[index_pos] = 1
            test_label[index_neg] = 0
            true_label = np.zeros(test_label.shape)
            true_label[0: len(true_label) // 2] = 1
            accuracy = accuracy_score(true_label, test_label)
            results.append(config.modes[i] + ":" + str(accuracy) + "\n")
    with open(config.result_filename, mode="a+") as f:
        f.writelines(results)


def sample(G, root, tree, sample_num, for_d):
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

    G.get_all_scores()
    all_score = G.all_scores
    all_score_ndarry = tlx.convert_to_numpy(all_score)
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
            relevance_probability = all_score_ndarry[current_node, node_neighbor]
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


def write_embeddings_to_file(G, D, n_node):
    # """write embeddings of the G and the D to files"""
    modes = [G, D]
    for i in range(2):
        embedding_matrix = modes[i].embedding_matrix
        index = np.array(range(n_node)).reshape(-1, 1)
        embedding_matrix = np.hstack([index, embedding_matrix])
        embedding_list = embedding_matrix.tolist()
        embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                         for emb in embedding_list]
        with open(config.emb_filenames[i], "w+") as f:
            lines = [str(n_node) + "\t" + str(config.n_emb) + "\n"] + embedding_str
            f.writelines(lines)


def get_node_pairs_from_path(path):
    """
    given a path from root to a sampled node, generate all the node pairs within the given windows size
    e.g., path = [1, 0, 2, 4, 2], window_size = 2 -->
    node pairs= [[1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]]
    :param path: a path from root to the sampled node
    :return pairs: a list of node pairs
    """

    path = path[:-1]
    pairs = []
    for i in range(len(path)):
        center_node = path[i]
        for j in range(max(i - config.window_size, 0), min(i + config.window_size + 1, len(path))):
            if i == j:
                continue
            node = path[j]
            pairs.append([center_node, node])
    return pairs


def prepare_data_for_d(G, root_nodes, graph, trees):
    """generate positive and negative samples for the D, and record them in the txt file"""

    center_nodes = []
    neighbor_nodes = []
    labels = []
    for i in root_nodes:
        if np.random.rand() < config.update_ratio:
            pos = graph[i]
            neg, _ = sample(G, i, trees[i], len(pos), for_d=True)
            if len(pos) != 0 and neg is not None:
                # positive samples
                center_nodes.extend([i] * len(pos))
                neighbor_nodes.extend(pos)
                labels.extend([1.0] * len(pos))

                # negative samples
                center_nodes.extend([i] * len(pos))
                neighbor_nodes.extend(neg)
                labels.extend([0.0] * len(neg))
    return center_nodes, neighbor_nodes, labels


def prepare_data_for_g(G, root_nodes, D, trees):
    """sample nodes for the G"""

    paths = []
    for i in root_nodes:
        if np.random.rand() < config.update_ratio:
            sample_nodes, paths_from_i = sample(G, i, trees[i], config.n_sample_gen, for_d=False)
            if paths_from_i is not None:
                paths.extend(paths_from_i)
    node_pairs = list(map(get_node_pairs_from_path, paths))
    node_1 = []
    node_2 = []
    for i in range(len(node_pairs)):
        for pair in node_pairs[i]:
            node_1.append(pair[0])
            node_2.append(pair[1])
    data = {
        "center_nodes": node_1,
        "neighbor_nodes": node_2
    }
    D.forward(data)
    reward = D.reward
    return node_1, node_2, reward


class Discriminator(Module):

    def __init__(self, n_node, node_emb_init):
        super(Discriminator, self).__init__()
        self.n_node = n_node
        self.node_emb_init = node_emb_init

        embedding = tlx.initializers.Constant(value=self.node_emb_init)
        invitor = tlx.initializers.Zeros()
        self.embedding_matrix = self._get_weights("b_embedding_matrix", shape=self.node_emb_init.shape,
                                                  init=embedding)
        self.bias_vector = self._get_weights("b_bias", shape=(self.n_node, 1), init=invitor)

    def forward(self, data):
        node_embedding = tlx.gather(self.embedding_matrix, data['center_nodes'])
        node_neighbor_embedding = tlx.gather(self.embedding_matrix, data['neighbor_nodes'])
        bias = tlx.gather(self.bias_vector, data['neighbor_nodes'])

        scores = tlx.nn.Reshape(shape=bias.shape)(
            tlx.reduce_sum(tlx.multiply(node_embedding, node_neighbor_embedding), axis=1)) + bias
        scores = tlx.clip_by_value(scores, clip_value_min=-10, clip_value_max=10)
        self.reward = tlx.log(1 + tlx.exp(scores))
        return node_embedding, node_neighbor_embedding, bias, scores


class Generator(Module):
    def __init__(self, n_node, node_emb_init):
        super(Generator, self).__init__()
        self.n_node = n_node
        self.node_emb_init = node_emb_init

        embedding = tlx.initializers.Constant(value=self.node_emb_init)
        invitor = tlx.initializers.Zeros()
        self.embedding_matrix = self._get_weights("b_embedding_matrix", shape=self.node_emb_init.shape,
                                                  init=embedding)
        self.bias_vector = self._get_weights("b_bias", shape=(self.n_node, 1), init=invitor)
        self.all_scores = tlx.matmul(self.embedding_matrix, self.embedding_matrix, transpose_b=True) + self.bias_vector

    def forward(self, data):
        node_embedding = tlx.gather(self.embedding_matrix, data['node_1'])
        node_neighbor_embedding = tlx.gather(self.embedding_matrix, data['node_2'])
        bias = tlx.gather(self.bias_vector, data['node_2'])
        score = tlx.nn.Reshape(shape=bias.shape)(
            tlx.reduce_sum(node_embedding * node_neighbor_embedding, axis=1)) + bias
        prob = tlx.clip_by_value(tlx.sigmoid(score), 1e-5, 1)
        return node_embedding, node_neighbor_embedding, prob

    def get_all_scores(self):
        self.all_scores = tlx.matmul(self.embedding_matrix, self.embedding_matrix, transpose_b=True) + self.bias_vector


class WithLossD(Module):

    def __init__(self, D):
        super(WithLossD, self).__init__()
        self.d_net = D

    def forward(self, data, label_sets):
        node_embedding, node_neighbor_embedding, bias, scores = self.d_net(data)
        zeros_embedding = tlx.ops.zeros(shape=node_neighbor_embedding.shape, dtype='float32')
        label_sets = tlx.nn.Reshape(shape=[labels_sets.shape.dims[0].value, 1])(label_sets)
        zeros_bias = tlx.ops.zeros(shape=bias.shape, dtype='float32')
        loss = tlx.reduce_sum(tlx.losses.sigmoid_cross_entropy(scores, label_sets)) + config.lambda_dis * (
                tlx.losses.mean_squared_error(node_embedding, zeros_embedding, 'sum') / 2 +
                tlx.losses.mean_squared_error(node_neighbor_embedding, zeros_embedding, 'sum') / 2 +
                tlx.losses.mean_squared_error(bias, zeros_bias, 'sum') / 2
        )
        return loss


class WithLossG(Module):

    def __init__(self, G):
        super(WithLossG, self).__init__()
        self.g_net = G

    def forward(self, data, reward_sets):
        node_embedding, node_neighbor_embedding, prob = self.g_net(data)
        zeros_embedding = tlx.ops.zeros(shape=node_neighbor_embedding.shape, dtype='float32')
        loss = -tlx.reduce_mean(tlx.log(prob) * reward_sets) + config.lambda_gen * (
                tlx.losses.mean_squared_error(node_neighbor_embedding, zeros_embedding, 'sum') / 2 +
                tlx.losses.mean_squared_error(node_embedding, zeros_embedding, 'sum') / 2
        )
        return loss


if __name__ == '__main__':

    n_node, graph = read_edges(config.train_filename, config.test_filename)
    root_nodes = [i for i in range(n_node)]

    node_embed_init_d = read_embeddings(filename=config.pretrain_emb_filename_d,
                                        n_node=n_node,
                                        n_embed=config.n_emb)
    node_embed_init_g = read_embeddings(filename=config.pretrain_emb_filename_g,
                                        n_node=n_node,
                                        n_embed=config.n_emb)
    trees = None
    if os.path.isfile(config.cache_filename):
        print("reading BFS-trees from cache...")
        pickle_file = open(config.cache_filename, 'rb')
        trees = pickle.load(pickle_file)
        pickle_file.close()
    else:
        print("constructing BFS-trees...")
        pickle_file = open(config.cache_filename, 'wb')
        if config.multi_processing:
            trees = {}
            construct_trees_with_mp(root_nodes, n_node, trees, graph)
        else:
            trees = construct_trees(graph, root_nodes)
        pickle.dump(trees, pickle_file)
        pickle_file.close()

    D = Discriminator(n_node, node_embed_init_d)
    G = Generator(n_node, node_embed_init_g)

    optimizer_d = tlx.optimizers.Adam(lr=config.lr_dis)
    optimizer_g = tlx.optimizers.Adam(lr=config.lr_gen)

    d_weights = D.trainable_weights
    g_weights = G.trainable_weights

    net_with_loss_D = WithLossD(D)
    net_with_loss_G = WithLossG(G)
    tranin_one_step_d = TrainOneStep(net_with_loss_D, optimizer_d, d_weights)
    tranin_one_step_g = TrainOneStep(net_with_loss_G, optimizer_g, g_weights)

    # restore the model from the latest checkpoint if exists
    d_num = len(glob.glob("checkpoint/*_d*.npz"))
    g_num = len(glob.glob("checkpoint/*_g*.npz"))
    if d_num and g_num:
        str_d = "-0000" + str(d_num)
        str_g = "-0000" + str(g_num)
        print(
            "loading the checkpoint: %s,%s" % ("model_d" + str_d + ".npz", "model_d" + str_g + ".npz"))
        tlx.files.load_and_assign_npz_dict(name="checkpoint/model_d" + str_d + ".npz", network=D)
        tlx.files.load_and_assign_npz_dict(name="checkpoint/model_g" + str_g + ".npz", network=G)

    write_embeddings_to_file(G, D, n_node)
    calculate_acc(n_node)

    for epoch in range(config.n_epochs):
        print("epoch %d" % epoch)

        # save the model
        # if epoch > 0 and epoch % config.save_steps == 0:
        #     tlx.files.save_npz(D.all_weights, name=config.model_log + 'model_d.npz')
        #     tlx.files.save_npz(G.all_weights, name=config.model_log + 'model_g.npz')

        # D-steps
        center_nodes = []
        neighbor_nodes = []
        labels = []
        for d_epoch in range(config.n_epochs_dis):
            # generate new nodes for the D for every dis_interval iterations
            if d_epoch % config.dis_interval == 0:
                center_nodes, neighbor_nodes, labels = prepare_data_for_d(G, root_nodes, graph, trees)
            # training
            train_size = len(center_nodes)
            start_list = list(range(0, train_size, config.batch_size_dis))
            np.random.shuffle(start_list)
            for start in start_list:
                end = start + config.batch_size_dis
                center_nodes_sets = tlx.convert_to_tensor(center_nodes[start:end])
                neighbor_nodes_sets = tlx.convert_to_tensor(neighbor_nodes[start:end])
                data = {
                    "center_nodes": center_nodes_sets,
                    "neighbor_nodes": neighbor_nodes_sets
                }
                labels_sets = tlx.convert_to_tensor(labels[start:end])
                tranin_one_step_d(data, labels_sets)

        # G-steps
        node_1 = []
        node_2 = []
        reward = []
        for g_epoch in range(config.n_epochs_gen):
            if g_epoch % config.gen_interval == 0:
                node_1, node_2, reward = prepare_data_for_g(G, root_nodes, D, trees)

            # training
            train_size = len(node_1)
            start_list = list(range(0, train_size, config.batch_size_gen))
            np.random.shuffle(start_list)
            for start in start_list:
                end = start + config.batch_size_gen
                node_1_sets = tlx.convert_to_tensor(node_1[start:end])
                node_2_sets = tlx.convert_to_tensor(node_2[start:end])
                reward_sets = tlx.convert_to_tensor(reward[start:end])
                data = {
                    "node_1": node_1_sets,
                    "node_2": node_2_sets,
                }
                tranin_one_step_g(data, reward_sets)

        store_num_d = len(glob.glob(pathname='checkpoint/*_d*.npz')) + 1
        store_num_g = len(glob.glob(pathname='checkpoint/*_g*.npz')) + 1
        stroe_path_d = 'checkpoint/model_d-0000' + str(store_num_d) + '.npz'
        stroe_path_g = 'checkpoint/model_g-0000' + str(store_num_g) + '.npz'
        tlx.files.save_npz_dict(save_list=D.all_weights, name=stroe_path_d)
        tlx.files.save_npz_dict(save_list=G.all_weights, name=stroe_path_g)

        write_embeddings_to_file(G, D, n_node)
        calculate_acc(n_node)
