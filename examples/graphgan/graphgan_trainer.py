# !/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'paddle'

import sys
sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
from gammagl.models import GraphGAN
from gammagl.utils import read_embeddings
from gammagl.datasets import CA_GrQc
import tensorlayerx as tlx
from sklearn.metrics import accuracy_score
from functools import partial
import numpy as np
import glob
import argparse
# tlx.set_device("GPU", 0)


def calculate_acc(n_node, test_edges, test_edges_neg, args):
    """
   Args:
       n_node:  number of nodes in the graph
       args: parameters setting

   Returns:
       (:obj:`float`, :obj:`float`):accuracy of generator and discriminator

   """
    edges = test_edges + test_edges_neg
    results = []
    d_val_acc = 0
    g_val_acc = 0

    emb_filenames = [f'{args.emb_folder}/CA-GrQc_gen_.emb',
                     f'{args.emb_folder}/CA-GrQc_dis_.emb']

    eval_filename = f'{args.eval_folder}/CA-GrQc.txt'

    for i in range(2):
        embed_filename = emb_filenames[i]
        n_emb = args.n_emb
        emd = read_embeddings(embed_filename, n_node=n_node, n_embed=n_emb)

        # may exists isolated point
        score_res = []
        for j in range(len(edges)):
            score_res.append(np.dot(emd[edges[j][0]], emd[edges[j][1]]))
        test_label = np.array(score_res)
        median = np.median(test_label)
        index_pos = test_label >= median
        index_neg = test_label < median
        test_label[index_pos] = 1
        test_label[index_neg] = 0
        true_label = np.zeros(test_label.shape)
        true_label[0: len(true_label) // 2] = 1
        accuracy = accuracy_score(true_label, test_label)
        if i == 0:
            g_val_acc = accuracy
        else:
            d_val_acc = accuracy
        results.append(args.modes[i] + ":" + str(accuracy) + "\n")

    # write accuracy of g and d to a file
    if not os.path.isdir(args.eval_folder):
        os.makedirs(args.eval_folder)

    with open(eval_filename, mode="a+") as f:
        f.writelines(results)
    return g_val_acc, d_val_acc


def prepare_data_for_d(GANModel, args):
    """
    Generate positive and negative samples for the discriminator, and record them in the txt file
    """
    center_nodes = []
    neighbor_nodes = []
    labels = []

    all_score = GANModel.generator.get_all_scores()
    all_score_ndarray = tlx.convert_to_numpy(all_score)

    for i in GANModel.root_nodes:
        if np.random.rand() < args.update_ratio:
            pos = GANModel.graph[i]
            neg, _ = GANModel.sample(all_score_ndarray, i,
                                     GANModel.trees[i], len(pos), for_d=True)
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


def prepare_data_for_g(GANModel, args):
    """
    Sample nodes for the G
    """
    paths = []
    all_score = GANModel.generator.get_all_scores()
    all_score_ndarray = tlx.convert_to_numpy(all_score)

    for i in GANModel.root_nodes:
        if np.random.rand() < args.update_ratio:
            sample_nodes, paths_from_i = GANModel.sample(
                all_score_ndarray, i, GANModel.trees[i], args.n_sample_gen, for_d=False)
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
    node1_ = tlx.convert_to_tensor(node_1)
    node2_ = tlx.convert_to_tensor(node_2)
    data = {
        "center_nodes": node1_,
        "neighbor_nodes": node2_,
        "nodes_num": len(node_1)
    }
    reward = GANModel.discriminator.get_reward(data)
    return node_1, node_2, reward


def get_node_pairs_from_path(args, path):
    """
    Given a path from root to a sampled node, generate all the node pairs within the given windows size
    e.g., path = [1, 0, 2, 4, 2], window_size = 2 -->
    node pairs= [[1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]]

    Args:
        args: parameters setting
        path: a path from root to the sampled node

    Returns:
        pairs: a list of node pairs

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


def write_embeddings_to_file(GANModel, args, choice):
    """write embeddings of the G and the D to files

    Args:
        GANModel:  GANModel model
        args: parameters setting
        choice:
            choice==1: write embeddings of G and D in files
            choice==2: write embeddings of G with the best accuracy in CA-GrQc_best_acc_gen_.emb
            choice==3: write embeddings of D with the best accuracy in CA-GrQc_best_acc_dis_.emb

    """
    modes = [GANModel.generator, GANModel.discriminator]
    emb_filenames = [f'{args.emb_folder}/CA-GrQc_gen_.emb',
                     f'{args.emb_folder}/CA-GrQc_dis_.emb']
    best_acc_emb_filenames = [f'{args.best_acc_emb_folder}/CA-GrQc_best_acc_gen_.emb',
                              f'{args.best_acc_emb_folder}/CA-GrQc_best_acc_dis_.emb']
    if choice == 1:
        for i in range(2):
            if tlx.BACKEND == 'torch':
                embedding_matrix = modes[i].embedding_matrix.detach().cpu()
            else:
                embedding_matrix = modes[i].embedding_matrix
            index = np.array(range(GANModel.n_node)).reshape(-1, 1)
            embedding_matrix = np.hstack([index, embedding_matrix])
            embedding_list = embedding_matrix.tolist()
            embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                             for emb in embedding_list]

            if not os.path.isdir(args.emb_folder):
                os.makedirs(args.emb_folder)

            with open(emb_filenames[i], "w+") as f:
                f.writelines(embedding_str)
    else:
        for i in range(2):
            if (choice == 2 and i == 0) or (choice == 3 and i == 1):
                if tlx.BACKEND == 'torch':
                    embedding_matrix = modes[i].embedding_matrix.detach().cpu()
                else:
                    embedding_matrix = modes[i].embedding_matrix
                index = np.array(range(GANModel.n_node)).reshape(-1, 1)
                embedding_matrix = np.hstack([index, embedding_matrix])
                embedding_list = embedding_matrix.tolist()
                embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                                 for emb in embedding_list]

                if not os.path.isdir(args.best_acc_emb_folder):
                    os.makedirs(args.best_acc_emb_folder)

                with open(best_acc_emb_filenames[i], "w+") as f:
                    f.writelines(embedding_str)


class WithLossD(tlx.nn.Module):

    def __init__(self, D):
        super(WithLossD, self).__init__()
        self.d_net = D

    def forward(self, data, label_sets):
        node_embedding, node_neighbor_embedding, bias, scores = self.d_net(
            data)
        label_sets = tlx.nn.Reshape(shape=[data['nodes_num'], 1])(label_sets)
        loss = tlx.reduce_sum(tlx.losses.sigmoid_cross_entropy(target=label_sets, output=scores)) + data[
            'args'].lambda_dis * (
                       tlx.reduce_sum(tlx.ops.square(node_embedding)) / 2 +
                       tlx.reduce_sum(tlx.ops.square(node_neighbor_embedding)) / 2 +
                       tlx.reduce_sum(tlx.ops.square(bias)) / 2
               )
        return loss


class WithLossG(tlx.nn.Module):

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
    dataset = CA_GrQc(args.dataset_path, args.n_emb)

    GANModel = GraphGAN(dataset.n_node, dataset.graph, dataset.node_embed_init_d,
                        dataset.node_embed_init_g, args.cache_folder, args.multi_processing)
    optimizer_d = tlx.optimizers.Adam(lr=args.lr_dis)
    optimizer_g = tlx.optimizers.Adam(lr=args.lr_gen)

    d_weights = GANModel.discriminator.trainable_weights
    g_weights = GANModel.generator.trainable_weights

    net_with_loss_D = WithLossD(GANModel.discriminator)
    net_with_loss_G = WithLossG(GANModel.generator)

    train_one_step_d = tlx.model.TrainOneStep(
        net_with_loss_D, optimizer_d, d_weights)
    train_one_step_g = tlx.model.TrainOneStep(
        net_with_loss_G, optimizer_g, g_weights)

    # Restore the model from the latest checkpoint if exists
    if args.load_model:
        d_num = len(glob.glob("checkpoint/*_d*.npz"))
        g_num = len(glob.glob("checkpoint/*_g*.npz"))
        if d_num and g_num:
            str_d = "-0000" + str(d_num)
            str_g = "-0000" + str(g_num)
            print(
                "loading the checkpoint: %s,%s" % ("model_d" + str_d + ".npz", "model_d" + str_g + ".npz"))
            GANModel.discriminator.load_weights(
                file_path="checkpoint/model_d" + str_d + ".npz", format='npz_dict')
            GANModel.generator.load_weights(
                file_path="checkpoint/model_g" + str_g + ".npz", format='npz_dict')

    write_embeddings_to_file(GANModel, args, 1)

    calculate_acc(GANModel.n_node, dataset.test_edges,
                  dataset.test_edges_neg, args)

    g_best_val_acc = 0
    d_best_val_acc = 0

    for epoch in range(args.n_epochs):
        # save the model
        if epoch > 0 and epoch % args.save_steps == 0:
            if not os.path.isdir('checkpoint'):
                os.makedirs('checkpoint')
            store_num_d = len(glob.glob(pathname='checkpoint/*_d*.npz')) + 1
            store_num_g = len(glob.glob(pathname='checkpoint/*_g*.npz')) + 1
            store_path_d = 'checkpoint/model_d-0000' + \
                           str(store_num_d) + '.npz'
            store_path_g = 'checkpoint/model_g-0000' + \
                           str(store_num_g) + '.npz'
            GANModel.discriminator.save_weights(
                file_path=store_path_d, format='npz_dict')
            GANModel.generator.save_weights(
                file_path=store_path_g, format='npz_dict')

        # D-steps
        center_nodes = []
        neighbor_nodes = []
        labels = []
        for d_epoch in range(args.n_epochs_dis):
            # Generate new nodes for the D for every dis_interval iterations
            if d_epoch % args.dis_interval == 0:
                center_nodes, neighbor_nodes, labels = prepare_data_for_d(
                    GANModel, args)

            # Training
            train_size = len(center_nodes)
            start_list = list(range(0, train_size, args.batch_size_dis))
            np.random.shuffle(start_list)
            for start in start_list:
                end = start + args.batch_size_dis
                center_nodes_sets = tlx.convert_to_tensor(
                    center_nodes[start:end])
                neighbor_nodes_sets = tlx.convert_to_tensor(
                    neighbor_nodes[start:end])
                data = {
                    "args": args,
                    "center_nodes": center_nodes_sets,
                    "neighbor_nodes": neighbor_nodes_sets,
                    "nodes_num": len(center_nodes_sets)
                }
                labels_sets = tlx.convert_to_tensor(labels[start:end])
                train_one_step_d(data, labels_sets)

        # G-steps
        node_1 = []
        node_2 = []
        for g_epoch in range(args.n_epochs_gen):
            if g_epoch % args.gen_interval == 0:
                # Generate new nodes for the G for every dis_interval iterations
                node_1, node_2, reward = prepare_data_for_g(GANModel, args)
                if tlx.BACKEND == 'paddle' or tlx.BACKEND == 'torch':
                    reward = reward.detach()
            # Training
            train_size = len(node_1)
            start_list = list(range(0, train_size, args.batch_size_gen))
            np.random.shuffle(start_list)
            for start in start_list:
                end = start + args.batch_size_gen
                node_1_sets = tlx.convert_to_tensor(node_1[start:end])
                node_2_sets = tlx.convert_to_tensor(node_2[start:end])
                reward_sets = reward[start:end]
                data = {
                    "args": args,
                    "node_1": node_1_sets,
                    "node_2": node_2_sets,
                }
                train_one_step_g(data, reward_sets)

        write_embeddings_to_file(GANModel, args, 1)
        g_val_acc, d_val_acc = calculate_acc(
            GANModel.n_node, dataset.test_edges, dataset.test_edges_neg, args)

        print(f'epoch : {epoch} \t generator acc : {g_val_acc:.4f} \t discriminator acc : {d_val_acc:.4f}')

        # Save best model on evaluation set
        if g_val_acc > g_best_val_acc:
            write_embeddings_to_file(GANModel, args, 2)
            g_best_val_acc = g_val_acc

        if d_val_acc > d_best_val_acc:
            write_embeddings_to_file(GANModel, args, 3)
            d_best_val_acc = d_val_acc

    # End training
    print("generator acc:  {:.4f}".format(g_best_val_acc))
    print("discriminator acc:  {:.4f}".format(d_best_val_acc))


if __name__ == '__main__':
    # Parameters setting
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument("--modes", type=list,
                        default=['gen', 'dis'],
                        help="modes names")
    parser.add_argument("--n_sample_gen", type=int, default=20,
                        help="number of samples for the generator")
    parser.add_argument("--update_ratio", type=int, default=1,
                        help="updating ratio when choose the trees")
    parser.add_argument("--batch_size_dis", type=int, default=1024,
                        help="batch size for the discriminator")
    parser.add_argument("--batch_size_gen", type=int, default=1024,
                        help="batch size for the generator")
    parser.add_argument("--n_epochs", type=int, default=30,
                        help="number of outer loops")
    parser.add_argument("--n_epochs_gen", type=int, default=30,
                        help="number of inner loops for the generator")
    parser.add_argument("--n_epochs_dis", type=int, default=30,
                        help="number of inner loops for the discriminator")
    parser.add_argument("--dis_interval", type=int, default=30,
                        help="sample new nodes for the discriminator for every dis_interval iterations")
    parser.add_argument("--gen_interval", type=int, default=30,
                        help="sample new nodes for the generator for every gen_interval iterations")
    parser.add_argument("--lambda_gen", type=float, default=1e-5,
                        help="l2 loss regulation weight for the generator")
    parser.add_argument("--lambda_dis", type=float, default=1e-5,
                        help="l2 loss regulation weight for the discriminator")
    parser.add_argument("--lr_gen", type=float, default=1e-5,
                        help="learning rate for the generator")
    parser.add_argument("--lr_dis", type=float, default=1e-5,
                        help="learning rate for the discriminator")

    # Model saving
    parser.add_argument("--load_model", type=bool, default=True,
                        help="whether loading existing model for initialization")
    parser.add_argument("--save_steps", type=int, default=10,
                        help="save steps")

    # Other hyper-parameters
    parser.add_argument("--n_emb", type=int, default=50,
                        help="the dimension of node embeddings")
    parser.add_argument("--window_size", type=int, default=2,
                        help="window size")
    parser.add_argument("--multi_processing", type=bool, default=True,
                        help="whether using multi-processing to construct BFS-trees")

    # Path settings
    parser.add_argument("--dataset_path", type=str,
                        default='gan_dataset', help="path to save dataset")
    parser.add_argument("--emb_folder", type=str,
                        default='gan_results',
                        help="embeddings during training filenames")
    parser.add_argument("--best_acc_emb_folder", type=str,
                        default='gan_results',
                        help="embeddings with the best accuracy filenames")
    parser.add_argument("--eval_folder", type=str,
                        default='gan_results',
                        help="evaluation result filename")
    parser.add_argument("--cache_folder", type=str,
                        default='gan_cache',
                        help="BFS-tree_cache_folder")

    args = parser.parse_args()
    main(args)
