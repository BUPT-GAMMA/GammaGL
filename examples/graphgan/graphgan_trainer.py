# !/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ['TL_BACKEND'] = 'paddle'

import sys

sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
import argparse
import numpy as np
import glob
import pickle
import collections
import multiprocessing
from functools import partial
import tqdm
from sklearn.metrics import accuracy_score

import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep
from tensorlayerx.nn import Module
from gammagl.utils import read_edges, read_embeddings, read_edges_from_file


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


def calculate_acc(n_node, args):
    results = []
    if args.app == "link_prediction":
        for i in range(2):
            test_filename = args.test_filename
            test_neg_filename = args.test_neg_filename
            embed_filename = args.emb_filenames[i]
            n_emb = args.n_emb
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
            results.append(args.modes[i] + ":" + str(accuracy) + "\n")
    with open(args.result_filename, mode="a+") as f:
        f.writelines(results)


def sample(all_score_ndarry, root, tree, sample_num, for_d):
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


def write_embeddings_to_file(G, D, n_node, args):
    # """write embeddings of the G and the D to files"""
    modes = [G, D]
    for i in range(2):
        embedding_matrix = modes[i].embedding_matrix
        index = np.array(range(n_node)).reshape(-1, 1)
        embedding_matrix = np.hstack([index, embedding_matrix])
        embedding_list = embedding_matrix.tolist()
        embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                         for emb in embedding_list]
        with open(args.emb_filenames[i], "w+") as f:
            lines = [str(n_node) + "\t" + str(args.n_emb) + "\n"] + embedding_str
            f.writelines(lines)


def get_node_pairs_from_path(args, path):
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
        for j in range(max(i - args.window_size, 0), min(i + args.window_size + 1, len(path))):
            if i == j:
                continue
            node = path[j]
            pairs.append([center_node, node])
    return pairs


def prepare_data_for_d(G, root_nodes, graph, trees, args):
    """generate positive and negative samples for the D, and record them in the txt file"""

    center_nodes = []
    neighbor_nodes = []
    labels = []

    G.get_all_scores()
    all_score = G.all_scores
    all_score_ndarry = tlx.convert_to_numpy(all_score)
    for i in root_nodes:
        if np.random.rand() < args.update_ratio:
            pos = graph[i]
            neg, _ = sample(all_score_ndarry, i, trees[i], len(pos), for_d=True)
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


def prepare_data_for_g(G, root_nodes, D, trees, args):
    """sample nodes for the G"""

    paths = []
    G.get_all_scores()
    all_score = G.all_scores
    all_score_ndarry = tlx.convert_to_numpy(all_score)
    for i in root_nodes:
        if np.random.rand() < args.update_ratio:
            sample_nodes, paths_from_i = sample(all_score_ndarry, i, trees[i], args.n_sample_gen, for_d=False)
            if paths_from_i is not None:
                paths.extend(paths_from_i)
    func = partial(get_node_pairs_from_path, args)
    node_pairs = list(map(func, paths))
    node_1 = []
    node_2 = []
    for i in range(len(node_pairs)):
        for pair in node_pairs[i]:
            node_1.append(pair[0])
            node_2.append(pair[1])
    data = {
        "center_nodes": node_1,
        "neighbor_nodes": node_2,
        "nodes_num": len(node_1)
    }
    reward = D.get_reward(data)
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
        return node_embedding, node_neighbor_embedding, bias, scores

    def get_reward(self, data):
        node_embedding = tlx.gather(self.embedding_matrix, data['center_nodes'])
        node_neighbor_embedding = tlx.gather(self.embedding_matrix, data['neighbor_nodes'])
        bias = tlx.gather(self.bias_vector, data['neighbor_nodes'])

        scores = tlx.nn.Reshape(shape=bias.shape)(
            tlx.reduce_sum(tlx.multiply(node_embedding, node_neighbor_embedding), axis=1)) + bias
        scores = tlx.clip_by_value(scores, clip_value_min=-10, clip_value_max=10)
        reward = tlx.log(1 + tlx.exp(scores))
        return reward


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
        label_sets = tlx.nn.Reshape(shape=[data['nodes_num'], 1])(label_sets)
        loss = tlx.reduce_sum(tlx.losses.sigmoid_cross_entropy(target=label_sets, output=scores)) + data[
            'args'].lambda_dis * (
                       tlx.reduce_sum(tlx.ops.square(node_embedding)) / 2 +
                       tlx.reduce_sum(tlx.ops.square(node_neighbor_embedding)) / 2 +
                       tlx.reduce_sum(tlx.ops.square(bias)) / 2
               )
        return loss


class WithLossG(Module):

    def __init__(self, G):
        super(WithLossG, self).__init__()
        self.g_net = G

    def forward(self, data, reward_sets):
        node_embedding, node_neighbor_embedding, prob = self.g_net(data)
        loss = -tlx.reduce_mean(tlx.log(prob) * reward_sets) + data['args'].lambda_gen * (
                tlx.reduce_sum(tlx.ops.square(node_embedding)) / 2 +
                tlx.reduce_sum(tlx.ops.square(node_neighbor_embedding)) / 2
        )
        return loss


def main(args):
    n_node, graph = read_edges(args.train_filename, args.test_filename)
    root_nodes = [i for i in range(n_node)]

    node_embed_init_d = read_embeddings(filename=args.pre_train_emb_filename_d,
                                        n_node=n_node,
                                        n_embed=args.n_emb)
    node_embed_init_g = read_embeddings(filename=args.pre_train_emb_filename_g,
                                        n_node=n_node,
                                        n_embed=args.n_emb)
    trees = None
    if os.path.isfile(args.cache_filename):
        print("reading BFS-trees from cache...")
        pickle_file = open(args.cache_filename, 'rb')
        trees = pickle.load(pickle_file)
        pickle_file.close()
    else:
        print("constructing BFS-trees...")
        pickle_file = open(args.cache_filename, 'wb')
        if args.multi_processing:
            trees = {}
            construct_trees_with_mp(root_nodes, n_node, trees, graph)
        else:
            trees = construct_trees(graph, root_nodes)
        pickle.dump(trees, pickle_file)
        pickle_file.close()

    D = Discriminator(n_node, node_embed_init_d)
    G = Generator(n_node, node_embed_init_g)

    optimizer_d = tlx.optimizers.Adam(lr=args.lr_dis)
    optimizer_g = tlx.optimizers.Adam(lr=args.lr_gen)

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

    write_embeddings_to_file(G, D, n_node, args)
    calculate_acc(n_node, args)

    for epoch in range(args.n_epochs):
        print("epoch %d" % epoch)

        # save the model
        # if epoch > 0 and epoch % config.save_steps == 0:
        #     tlx.files.save_npz(D.all_weights, name=config.model_log + 'model_d.npz')
        #     tlx.files.save_npz(G.all_weights, name=config.model_log + 'model_g.npz')

        # D-steps
        center_nodes = []
        neighbor_nodes = []
        labels = []
        for d_epoch in range(args.n_epochs_dis):
            print("d_epoch %d" % d_epoch)
            # generate new nodes for the D for every dis_interval iterations
            if d_epoch % args.dis_interval == 0:
                center_nodes, neighbor_nodes, labels = prepare_data_for_d(G, root_nodes, graph, trees, args)
            # training
            train_size = len(center_nodes)
            start_list = list(range(0, train_size, args.batch_size_dis))
            np.random.shuffle(start_list)
            for start in start_list:
                end = start + args.batch_size_dis
                center_nodes_sets = tlx.convert_to_tensor(center_nodes[start:end])
                neighbor_nodes_sets = tlx.convert_to_tensor(neighbor_nodes[start:end])
                data = {
                    "args": args,
                    "center_nodes": center_nodes_sets,
                    "neighbor_nodes": neighbor_nodes_sets,
                    "nodes_num": len(center_nodes_sets)
                }
                labels_sets = tlx.convert_to_tensor(labels[start:end])
                tranin_one_step_d(data, labels_sets)

        # G-steps
        node_1 = []
        node_2 = []
        for g_epoch in range(args.n_epochs_gen):
            print("g_epoch %d" % g_epoch)
            if g_epoch % args.gen_interval == 0:
                node_1, node_2, reward = prepare_data_for_g(G, root_nodes, D, trees, args)

            # trainin
            train_size = len(node_1)
            start_list = list(range(0, train_size, args.batch_size_gen))
            np.random.shuffle(start_list)
            for start in start_list:
                end = start + args.batch_size_gen
                node_1_sets = node_1[start:end]
                node_2_sets = node_2[start:end]
                reward_sets = reward[start:end]
                data = {
                    "args": args,
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

        write_embeddings_to_file(G, D, n_node, args)
        calculate_acc(n_node, args)


if __name__ == '__main__':
    # application and dataset settings
    app = "link_prediction"
    dataset = "CA-GrQc"

    # parameters setting
    parser = argparse.ArgumentParser()

    # training settings
    parser.add_argument("--app", type=str, default='link_prediction',
                        help="app_name")
    parser.add_argument("--modes", type=list,
                        default=['gen', 'dis'],
                        help="modes_names")
    parser.add_argument("--n_sample_gen", type=int, default=20,
                        help="number of samples for the generator")
    parser.add_argument("--update_ratio", type=int, default=1,
                        help="updating ratio when choose the trees")
    parser.add_argument("--batch_size_dis", type=int, default=64,
                        help="batch size for the discriminator")
    parser.add_argument("--batch_size_gen", type=int, default=64,
                        help="batch size for the generator")
    parser.add_argument("--n_epochs", type=int, default=20, help="number of outer loops")
    parser.add_argument("--n_epochs_gen", type=int, default=30, help="number of inner loops for the generator")
    parser.add_argument("--n_epochs_dis", type=int, default=30, help="number of inner loops for the discriminator")
    parser.add_argument("--dis_interval", type=int, default=30,
                        help="sample new nodes for the discriminator for every dis_interval iterations")
    parser.add_argument("--gen_interval", type=int, default=30,
                        help="sample new nodes for the generator for every gen_interval iterations")
    parser.add_argument("--lambda_gen", type=float, default=1e-5, help="l2 loss regulation weight for the generator")
    parser.add_argument("--lambda_dis", type=float, default=1e-5,
                        help="l2 loss regulation weight for the discriminator")
    parser.add_argument("--lr_gen", type=float, default=1e-3, help="learning rate for the generator")
    parser.add_argument("--lr_dis", type=float, default=1e-3, help="learning rate for the discriminator")

    # model saving
    parser.add_argument("--load_model", type=bool, default=False,
                        help="whether loading existing model for initialization")
    parser.add_argument("--save_steps", type=int, default=10,
                        help="save_steps")

    # other hyper-parameters
    parser.add_argument("--n_emb", type=int, default=50, help="hyper_parameters_n_emb")
    parser.add_argument("--window_size", type=int, default=2,
                        help="window_size")
    parser.add_argument("--multi_processing", type=bool, default=True,
                        help="whether using multi-processing to construct BFS-trees")

    # path settings
    parser.add_argument("--train_filename", type=str, default='gan_datasets/' + app + '/' + dataset + '_train.txt',
                        help="train_filename")
    parser.add_argument("--pre_train_emb_filename_d", type=str,
                        default='gan_datasets/' + app + '/' + dataset + '_pre_train.emb',
                        help="pre_train_d_emb_filename")
    parser.add_argument("--pre_train_emb_filename_g", type=str,
                        default='gan_datasets/' + app + '/' + dataset + '_pre_train.emb',
                        help="pre_train_g_emb_filename")
    parser.add_argument("--test_filename", type=str, default='gan_datasets/' + app + '/' + dataset + '_test.txt',
                        help="test_filename")
    parser.add_argument("--test_neg_filename", type=str,
                        default='gan_datasets/' + app + '/' + dataset + '_test_neg.txt',
                        help="test_neg_filename")
    parser.add_argument("--emb_filenames", type=list,
                        default=['gan_results/' + app + '/' + dataset + '_gen_.emb',
                                 'gan_results/' + app + '/' + dataset + '_dis_.emb'],
                        help="emb_filenames")
    parser.add_argument("--result_filename", type=str,
                        default='gan_results/' + app + '/' + dataset + '.txt',
                        help="result_filename")
    parser.add_argument("--cache_filename", type=str,
                        default='gan_cache/' + dataset + '.pkl',
                        help="BFS-tree_cache_filename")

    args = parser.parse_args()
    main(args)
