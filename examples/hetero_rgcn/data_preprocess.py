import os
os.environ["TL_BACKEND"] = "torch"
import argparse

import tensorlayerx as tlx
import numpy as np
from tqdm import tqdm
import os.path as osp


from gammagl.data import HeteroGraph

edge_size = 0
node_size = 0


def read_node_atts(node_file, label_file=None):
    node_maps = {}
    node_embeds = {}
    count = 0
    lack_num = {}
    node_counts = node_size

    print("Start loading node information")
    process = tqdm(total=node_counts)
    with open(node_file, 'r') as rf:
        while True:
            line = rf.readline()
            if line is None or len(line) == 0:
                break
            info = line.strip().split(",")

            node_id = int(info[0])
            node_type = info[1].strip()

            node_maps.setdefault(node_type, {})
            node_id_v2 = len(node_maps[node_type])
            node_maps[node_type][node_id] = node_id_v2

            node_embeds.setdefault(node_type, {})
            lack_num.setdefault(node_type, 0)
            if node_type == 'item':
                if len(info[2]) < 50:
                    node_embeds[node_type][node_id_v2] = np.zeros(256, dtype=np.float32)
                    lack_num[node_type] += 1
                else:
                    node_embeds[node_type][node_id_v2] = np.array([x for x in info[2].split(":")], dtype=np.float32)
            else:
                if len(info[2]) < 50:
                    node_embeds[node_type][node_id_v2] = np.zeros(256, dtype=np.float32)
                    lack_num[node_type] += 1
                else:
                    node_embeds[node_type][node_id_v2] = np.array([x for x in info[2].split(":")], dtype=np.float32)

            count += 1
            if count % 100000 == 0:
                process.update(100000)

    process.update(node_size % 100000)
    process.close()
    print("Complete loading node information\n")

    print("Num of total nodes:", count)
    print('Node_types:', list(node_maps.keys()))
    print('Node_type Num Num_lack_feature:')
    for node_type in node_maps:
        print(node_type, len(node_maps[node_type]), lack_num[node_type])

    labels = []
    if label_file is not None:
        labels_info = [x.strip().split(",") for x in open(label_file).readlines()]
        for i in range(len(labels_info)):
            x = labels_info[i]
            item_id = node_maps['item'][int(x[0])]
            label = int(x[1])
            labels.append([item_id, label])

    nodes_dict = {'maps': node_maps, 'embeds': node_embeds}
    nodes_dict['labels'] = {}
    nodes_dict['labels']['item'] = labels
    print('\n')
    print('Finish loading node information')

    graph = HeteroGraph()

    print("Start converting into Gammagl data")
    for node_type in tqdm(node_embeds, desc="Node features, numbers and mapping"):

        graph[node_type].x = np.zeros((len(node_maps[node_type]), 256), dtype=np.float32)
        # graph[node_type].x = tlx.convert_to_tensor(node_embeds[node_type].items())
        for nid, embedding in tqdm(node_embeds[node_type].items()):
            graph[node_type].x[nid] = embedding
        graph[node_type].x = tlx.convert_to_tensor(graph[node_type].x)
        graph[node_type].num_nodes = len(node_maps[node_type])
        graph[node_type].maps = node_maps[node_type]

    if label_file is not None:
        graph['item'].y = np.zeros((len(node_maps['item']), ), dtype=np.int64) - 1
        for index, label in tqdm(labels, desc="Node labels"):
            graph['item'].y[index] = label
        # graph['item'].y = tlx.convert_to_tensor(graph['item'].y)
        indices = (graph['item'].y != -1).nonzero()[0]
        print("Num of true labeled nodes:{}".format(indices.shape[0]))
        train_val_random = np.random.permutation(indices.shape[0])
        train_idx = indices[train_val_random][:int(indices.shape[0] * 0.8)]
        val_idx = indices[train_val_random][int(indices.shape[0] * 0.8):]
        graph['item'].train_idx = tlx.convert_to_tensor(train_idx, tlx.int64)
        graph['item'].val_idx = tlx.convert_to_tensor(val_idx, tlx.int64)
        graph['item'].y = tlx.convert_to_tensor(graph['item'].y)
    for ntype in graph.node_types:
        graph[ntype].n_id = tlx.arange(0, graph[ntype].num_nodes)
    print("Complete converting into GammaGL data\n")

    return graph


def format_ggl_graph(edge_file, node_file, ggl_file, label_file=None):
    if tlx.BACKEND == "torch":
        import torch
        if osp.exists(ggl_file + "icdm2022_torch.pt"):
            return 0
        else:
            print("##########################################")
            print("### Start generating GammaGL torch graph")
            print("##########################################\n")
            graph = read_node_atts(node_file, label_file)

        print("Start loading edge information")
        process = tqdm(total=edge_size)
        edges = {}
        count = 0
        with open(edge_file, 'r') as rf:
            while True:
                line = rf.readline()
                if line is None or len(line) == 0:
                    break
                line_info = line.strip().split(",")
                source_id, dest_id, source_type, dest_type, edge_type = line_info
                source_id = graph[source_type].maps[int(source_id)]
                dest_id = graph[dest_type].maps[int(dest_id)]
                edges.setdefault(edge_type, {})
                edges[edge_type].setdefault('source', []).append(int(source_id))
                edges[edge_type].setdefault('dest', []).append(int(dest_id))
                edges[edge_type].setdefault('source_type', source_type)
                edges[edge_type].setdefault('dest_type', dest_type)
                count += 1
                if count % 100000 == 0:
                    process.update(100000)
        process.update(edge_size % 100000)
        process.close()
        print('Complete loading edge information\n')

        print('Start converting edge information')
        for edge_type in edges:
            source_type = edges[edge_type]['source_type']
            dest_type = edges[edge_type]['dest_type']
            source = tlx.convert_to_tensor(edges[edge_type]['source'], dtype=tlx.int64)
            dest = tlx.convert_to_tensor(edges[edge_type]['dest'], dtype=tlx.int64)
            graph[(source_type, edge_type, dest_type)].edge_index = tlx.stack([source, dest])

        for edge_type in [('b', 'A_1', 'item'),
                          ('f', 'B', 'item'),
                          ('a', 'G_1', 'f'),
                          ('f', 'G', 'a'),
                          ('a', 'H_1', 'e'),
                          ('f', 'C', 'd'),
                          ('f', 'D', 'c'),
                          ('c', 'D_1', 'f'),
                          ('f', 'F', 'e'),
                          ('item', 'B_1', 'f'),
                          ('item', 'A', 'b'),
                          ('e', 'F_1', 'f'),
                          ('e', 'H', 'a'),
                          ('d', 'C_1', 'f')]:
            temp = graph[edge_type].edge_index
            del graph[edge_type]
            graph[edge_type].edge_index = temp

        print('Complete converting edge information\n')
        print('Start saving into gammagl data')
        # 这加一个判断backend，然后对应生成.pt类的文件
        path = ggl_file + "icdm2022_torch.pt"
        torch.save(graph, path)
        print('Complete saving into ggl data\n')

        print("##########################################")
        print("### Complete generating GammaGL torch graph")
        print("##########################################")
    elif tlx.BACKEND == "tensorflow":
        import pickle
        if osp.exists(ggl_file + "icdm2022_tensorflow.pt"):
            return 0
        else:
            print("##########################################")
            print("### Start generating GammaGL tensorflow graph")
            print("##########################################\n")
            graph = read_node_atts(node_file, label_file)
        print("Start loading edge information")
        process = tqdm(total=edge_size)
        edges = {}
        count = 0
        with open(edge_file, 'r') as rf:
            while True:
                line = rf.readline()
                if line is None or len(line) == 0:
                    break
                line_info = line.strip().split(",")
                source_id, dest_id, source_type, dest_type, edge_type = line_info
                source_id = graph[source_type].maps[int(source_id)]
                dest_id = graph[dest_type].maps[int(dest_id)]
                edges.setdefault(edge_type, {})
                edges[edge_type].setdefault('source', []).append(int(source_id))
                edges[edge_type].setdefault('dest', []).append(int(dest_id))
                edges[edge_type].setdefault('source_type', source_type)
                edges[edge_type].setdefault('dest_type', dest_type)
                count += 1
                if count % 100000 == 0:
                    process.update(100000)
        process.update(edge_size % 100000)
        process.close()
        print('Complete loading edge information\n')

        print('Start converting edge information')
        for edge_type in edges:
            source_type = edges[edge_type]['source_type']
            dest_type = edges[edge_type]['dest_type']
            source = tlx.convert_to_tensor(edges[edge_type]['source'], dtype=tlx.int64)
            dest = tlx.convert_to_tensor(edges[edge_type]['dest'], dtype=tlx.int64)
            graph[(source_type, edge_type, dest_type)].edge_index = tlx.stack([source, dest])

        for edge_type in [('b', 'A_1', 'item'),
                          ('f', 'B', 'item'),
                          ('a', 'G_1', 'f'),
                          ('f', 'G', 'a'),
                          ('a', 'H_1', 'e'),
                          ('f', 'C', 'd'),
                          ('f', 'D', 'c'),
                          ('c', 'D_1', 'f'),
                          ('f', 'F', 'e'),
                          ('item', 'B_1', 'f'),
                          ('item', 'A', 'b'),
                          ('e', 'F_1', 'f'),
                          ('e', 'H', 'a'),
                          ('d', 'C_1', 'f')]:
            temp = graph[edge_type].edge_index
            del graph[edge_type]
            graph[edge_type].edge_index = temp

        print('Complete converting edge information\n')
        print('Start saving into gammagl data')
        # 这加一个判断backend，然后对应生成.pt类的文件
        path = ggl_file + "icdm2022_tensorflow.pt"
        with open(path, 'wb') as f:
            graph.numpy()
            pickle.dump(graph, f)
        print('Complete saving into ggl data\n')

        print("##########################################")
        print("### Complete generating GammaGL tensorflow graph")
        print("##########################################")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--graph', type=str, default="./icdm2022_small/icdm2022_edges.csv")
    # parser.add_argument('--node', type=str, default="./icdm2022_small/icdm2022_nodes.csv")
    # parser.add_argument('--label', type=str, default="./icdm2022_small/icdm2022_labels.csv")
    parser.add_argument('--graph', type=str, default="./icdm2022_session1/icdm2022_session1_edges.csv")
    parser.add_argument('--node', type=str, default="./icdm2022_session1/icdm2022_session1_nodes.csv")
    parser.add_argument('--label', type=str, default="./icdm2022_session1/icdm2022_session1_train_labels.csv")
    parser.add_argument('--storefile', type=str, default='./zsy_test/')
    parser.add_argument('--reload', type=bool, default=False, help="Whether node features should be reloaded")


    args = parser.parse_args()

    if args.graph is not None and args.storefile is not None and args.node is not None:
        format_ggl_graph(args.graph, args.node, args.storefile, args.label)
        # read_node_atts(args.node, args.storefile, args.label)
