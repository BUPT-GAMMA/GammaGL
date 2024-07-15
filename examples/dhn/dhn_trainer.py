import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR

import argparse
import random
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep
from sklearn.metrics import roc_auc_score
from gammagl.models import DHNModel
from gammagl.datasets import ACM4DHN
from gammagl.utils import k_hop_subgraph, find_all_simple_paths


type2idx = {
    'M': 0,
    'A': 1,
    # 'C': 2,
    # 'T': 3
}


def dist_encoder(src, dest, G, k_hop):
    if (G.size(1) == 0):
        paths = []
    else:
        paths = find_all_simple_paths(G, src, dest, k_hop + 2)

    node_type = len(type2idx)
    cnt = [k_hop + 1] * node_type  # Default truncation for max_spd exceeded
    for path in paths:
        res = [0] * node_type
        for i in path:
            if i >= 0:
                res[type2idx['M']] += 1
            else:
                res[type2idx['A']] += 1

        for k in range(node_type):
            cnt[k] = min(cnt[k], res[k])

    # Generate one-hot encoding
    if args.one_hot:
        one_hot_list = [np.eye(k_hop + 2, dtype=np.float64)[cnt[i]]
                        for i in range(node_type)]
        return np.concatenate(one_hot_list)
    return cnt


def type_encoder(node):
    node_type = len(type2idx)
    res = [0] * node_type
    if node.item() >= 0:
        res[type2idx['M']] = 1.0
    else:
        res[type2idx['A']] = 1.0
    return res


mini_batch = []
fea_batch = []


def gen_fea_batch(G, root, fea_dict, k_hop):
    fea_batch = []
    mini_batch.append(root)

    a = [0] * (k_hop + 2) * 4 + type_encoder(root)

    node_type = len(type2idx)
    num_fea = (k_hop + 2) * 4 + node_type
    fea_batch.append(np.asarray(a,
                                dtype=np.float32
                                ).reshape(-1, num_fea)
                     )

    # 1-order neighbor sampling
    ns_1 = []
    src, dst = G
    for node in mini_batch[-1]:
        if node.item() >= 0:
            neighbors_mask = src == node
        else:
            neighbors_mask = dst == node
        neighbors = list(tlx.convert_to_numpy(dst[neighbors_mask]))
        neighbors.append(node.item())
        random_choice_list = np.random.choice(neighbors, args.num_neighbor, replace=True)
        ns_1.append(random_choice_list.tolist())
    ns_1 = tlx.convert_to_tensor(ns_1)
    mini_batch.append(ns_1[0])

    de_1 = [
        np.concatenate([fea_dict[ns_1[0][i].item()], np.asarray(type_encoder(ns_1[0][i]))], axis=0)
        for i in range(0, ns_1[0].shape[0])
    ]

    fea_batch.append(np.asarray(de_1,
                                dtype=np.float32).reshape(1, -1)
                     )

    # 2-order neighbor sampling
    ns_2 = []
    for node in mini_batch[-1]:
        if node.item() >= 0:
            neighbors_mask = src == node
        else:
            neighbors_mask = dst == node
        neighbors = list(tlx.convert_to_numpy(dst[neighbors_mask]))
        neighbors.append(node.item())
        random_choice_list = np.random.choice(neighbors, args.num_neighbor, replace=True)
        ns_2.append(random_choice_list.tolist())
    ns_2 = tlx.convert_to_tensor(ns_2)

    de_2 = []
    for i in range(len(ns_2)):
        tmp = []
        for j in range(len(ns_2[0])):
            tmp.append(
                np.concatenate([fea_dict[ns_2[i][j].item()], np.asarray(type_encoder(ns_2[i][j]))], axis=0)
            )
        de_2.append(tmp)

    fea_batch.append(np.asarray(de_2,
                                dtype=np.float32).reshape(1, -1)
                     )

    return np.concatenate(fea_batch, axis=1)


def subgraph_sampling_with_DE_node_pair(G, node_pair, k_hop=2):
    [A, B] = node_pair

    edge_index = tlx.concat([G['M', 'MA', 'A'].edge_index, reversed(G['M', 'MA', 'A'].edge_index)], axis=1)

    # Find k-hop subgraphs of A and B
    sub_G_for_AB = k_hop_subgraph([A, B], k_hop, edge_index)

    # Remove edges using Boolean indexes
    # Note: Just remove the edges, the points remain
    edge_index_np = tlx.convert_to_numpy(sub_G_for_AB[1])
    remove_indices = tlx.convert_to_tensor([
        ((edge_index_np[0, i] == A) & (edge_index_np[1, i] == B)) | (
                (edge_index_np[0, i] == B) & (edge_index_np[1, i] == A))
        for i in range(sub_G_for_AB[1].shape[1])
    ])
    remove_indices = tlx.convert_to_numpy(remove_indices)
    sub_G_index = sub_G_for_AB[1][:, ~remove_indices]

    sub_G_nodes = set(np.unique(tlx.convert_to_numpy(sub_G_for_AB[0]))) | set(
        np.unique(tlx.convert_to_numpy(sub_G_for_AB[1])))  # Gets the points in the graph
    sub_G_nodes = tlx.convert_to_tensor(list(sub_G_nodes))

    # Distance from all points in the subgraph to the node pair
    SPD_based_on_node_pair = {}
    for node in sub_G_nodes:
        tmpA = dist_encoder(A, node, sub_G_index, k_hop)
        tmpB = dist_encoder(B, node, sub_G_index, k_hop)

        SPD_based_on_node_pair[node.item()] = np.concatenate([tmpA, tmpB], axis=0)

    A_fea_batch = gen_fea_batch(sub_G_index, A,
                                SPD_based_on_node_pair, k_hop)
    B_fea_batch = gen_fea_batch(sub_G_index, B,
                                SPD_based_on_node_pair, k_hop)

    return A_fea_batch, B_fea_batch


def batch_data(G, batch_size):
    edge_index = G['M', 'MA', 'A'].edge_index
    nodes = set(tlx.convert_to_tensor(np.unique(tlx.convert_to_numpy(edge_index[0])))) | set(
        tlx.convert_to_tensor(np.unique(tlx.convert_to_numpy(edge_index[1]))))

    nodes_list = []
    for node in nodes:
        nodes_list.append(node.item())

    num_batch = int(len(edge_index[0]) / batch_size)

    # Shuffle the order of the edges
    edge_index_np = tlx.convert_to_numpy(edge_index)
    permutation = np.random.permutation(edge_index_np.shape[1])  # Generate a randomly arranged index
    edge_index_np = edge_index_np[:, permutation]  # Use this permutation index to scramble edge_index
    edge_index = tlx.convert_to_tensor(edge_index_np)

    for idx in range(num_batch):
        batch_edge = edge_index[:, idx * batch_size:(idx + 1) * batch_size]  # Take out batch_size edges
        batch_label = [1.0] * batch_size

        batch_A_fea = []
        batch_B_fea = []
        batch_x = []
        batch_y = []

        i = 0
        for by in batch_label:
            bx = batch_edge[:, i:i + 1]

            # Positive sample
            posA, posB = subgraph_sampling_with_DE_node_pair(G, bx, k_hop=args.k_hop)
            batch_A_fea.append(posA)
            batch_B_fea.append(posB)
            batch_y.append(np.asarray(by, dtype=np.float32))

            # Negative sample
            neg_tmpB_id = random.choice(nodes_list)
            node_pair = tlx.convert_to_tensor([[bx[0].item()], [neg_tmpB_id]])

            negA, negB = subgraph_sampling_with_DE_node_pair(G, node_pair, k_hop=args.k_hop)
            batch_A_fea.append(negA)
            batch_B_fea.append(negB)
            batch_y.append(np.asarray(0.0, dtype=np.float32))

        yield np.asarray(np.squeeze(batch_A_fea)), np.asarray(np.squeeze(batch_B_fea)), np.asarray(
            batch_y).reshape(batch_size * 2, 1)


class Loss(tlx.model.WithLoss):
    def __init__(self, net, loss_fn):
        super(Loss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['n1'], data['n2'], data['label'])
        y = tlx.convert_to_tensor(y)
        loss = self._loss_fn(logits, y)
        return loss


class AUCMetric:
    def __init__(self):
        self.true_labels = []
        self.predicted_scores = []

    def update_state(self, y_true, y_pred):
        self.true_labels.extend(y_true)
        self.predicted_scores.extend(y_pred)

    def result(self):
        auc = roc_auc_score(self.true_labels, self.predicted_scores)
        return auc


def main(args):
    if str.lower(args.dataset) not in ['acm']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    if str.lower(args.dataset) == 'acm':
        data = ACM4DHN(root=args.dataset_path, test_ratio=args.test_ratio)

    graph = data[0]

    G_train = graph['train']
    G_val = graph['val']
    G_test = graph['test']

    node_type = len(type2idx)
    num_fea = (args.k_hop + 2) * 4 + node_type

    model = DHNModel(num_fea, args.batch_size, args.num_neighbor, name="DHN")

    optim = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.drop_rate)
    train_weights = model.trainable_weights

    net_with_loss = Loss(model, loss_fn=tlx.losses.sigmoid_cross_entropy)
    net_with_train = TrainOneStep(net_with_loss, optim, train_weights)

    tra_auc_metric = AUCMetric()
    val_auc_metric = AUCMetric()
    test_auc_metric = AUCMetric()

    best_val_auc = 0
    for epoch in range(args.n_epoch):

        # train
        model.set_train()
        tra_batch_A_fea, tra_batch_B_fea, tra_batch_y = batch_data(G_train, args.batch_size).__next__()
        tra_out = model(tra_batch_A_fea, tra_batch_B_fea, tra_batch_y)

        data = {
            "n1": tra_batch_A_fea,
            "n2": tra_batch_B_fea,
            "label": tra_batch_y
        }

        tra_loss = net_with_train(data, tra_batch_y)
        tra_auc_metric.update_state(y_true=tra_batch_y, y_pred=tlx.convert_to_numpy(tlx.sigmoid(tra_out)))
        tra_auc = tra_auc_metric.result()

        # val
        model.set_eval()
        val_batch_A_fea, val_batch_B_fea, val_batch_y = batch_data(G_val, args.batch_size).__next__()
        val_out = model(val_batch_A_fea, val_batch_B_fea, val_batch_y)

        val_auc_metric.update_state(y_true=val_batch_y, y_pred=tlx.convert_to_numpy(tlx.sigmoid(val_out)))
        val_auc = val_auc_metric.result()

        print("Epoch [{:0>3d}] ".format(epoch+1)\
              + "  train loss: {:.4f}".format(tra_loss.item())\
              + "  val auc: {:.4f}".format(val_auc))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            model.save_weights(args.best_model_path+model.name+".npz", format='npz_dict')

    model.load_weights(args.best_model_path+model.name+".npz", format='npz_dict')
    # test
    test_batch_A_fea, test_batch_B_fea, test_batch_y = batch_data(G_test, args.batch_size).__next__()
    test_out = model(test_batch_A_fea, test_batch_B_fea, test_batch_y)

    test_auc_metric.update_state(y_true=test_batch_y, y_pred=tlx.convert_to_numpy(tlx.sigmoid(test_out)))
    test_auc = test_auc_metric.result()
    print("Test auc:  {:.4f}".format(test_auc))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_ratio", type=float, default=0.3, help="ratio of dividing the data set")
    parser.add_argument("--one_hot", type=bool, default=True, help="use one-hot encoding")
    parser.add_argument("--k_hop", type=int, default=2, help="hops of the generated subgraph")
    parser.add_argument("--num_neighbor", type=int, default=5, help="neighbor sample number")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--n_epoch", type=int, default=100, help="number of epoch")
    parser.add_argument("--drop_rate", type=float, default=0.01, help="drop_rate")
    parser.add_argument('--dataset', type=str, default='acm', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r"", help='dataset_path')
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--gpu", type=int, default=-1)

    args = parser.parse_args()
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    main(args)
