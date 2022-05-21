import networkx as nx
import numpy as np
import scipy as sc
import os
import re
import tensorlayerx as tlx
# import torch

import pickle
import random

from gammagl.data.graph_sampler import GraphSampler

#import util
# ---- NetworkX compatibility
def node_iter(G):
    # if float(nx.__version__)<2.0:
    #     return G.nodes()
    # else:
    return G.nodes

def node_dict(G):
    #if float(nx.__version__)>2.1:
    node_dict = G.nodes #159
    # else:
    #     node_dict = G.node
    return node_dict
# ---------------------------

def exp_moving_avg(x, decay=0.9):
    shadow = x[0]
    a = [shadow]
    for v in x[1:]:
        shadow -= (1-decay) * (shadow-v)
        a.append(shadow)
    return a

def read_graphfile(datadir, dataname, max_nodes=None):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')

    filename_node_attrs = prefix + '_node_attributes.txt'
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')

    label_has_zero = False
    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []

    # assume that all graph labels appear in the dataset
    # (set of labels don't have to be consecutive)
    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            # if val == 0:
            #    label_has_zero = True
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    # graph_labels = np.array(graph_labels)
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
    # if label_has_zero:
    #    graph_labels += 1

    filename_adj = prefix + '_A.txt'
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))
            index_graph[graph_indic[e0]] += [e0, e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs = []
    for i in range(1, 1 + len(adj_list)):
        # indexed from 1 here
        G = nx.from_edgelist(adj_list[i])
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue

        # add features and labels
        G.graph['label'] = graph_labels[i - 1]
        for u in node_iter(G):
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                node_label_one_hot[node_label] = 1
                node_dict(G)[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                node_dict(G)[u]['feat'] = node_attrs[u - 1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        # relabeling
        mapping = {}
        it = 0
        for n in node_iter(G):
            mapping[n] = it
            it += 1

        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))
    return graphs


def prepare_val_data(graphs, args, val_idx, max_nodes=0):
# DD 1168
    random.shuffle(graphs)
    val_size = len(graphs) // 10 #116
    train_graphs = graphs[:val_idx * val_size] #1052
    if val_idx < 9:
        train_graphs = train_graphs + graphs[(val_idx+1) * val_size :]
    val_graphs = graphs[val_idx*val_size: (val_idx+1)*val_size] #116
    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

# #Using node labels
# Num training graphs:  1052 ; Num validation graphs:  116
# Number of graphs:  1168
# Number of edges:  789819
#Max, avg, std of graph size:  903 , 268.70 , 161.33

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type) #adj_all1052; feat_all 1052; assign_feat_dim 1052; feat_dim 89

    train_dataset_loader = tlx.dataflow.DataLoader(
            dataset_sampler,
            batch_size=args.batch_size, #20 #batch_sampler53
            shuffle=True,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type) #116,89
    val_dataset_loader = tlx.dataflow.DataLoader(
            dataset_sampler,
            batch_size=args.batch_size, #20 #6
            shuffle=False,
            num_workers=args.num_workers)
#53,6
    return train_dataset_loader, val_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim #89.89

