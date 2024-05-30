import random
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep
import tensorflow as tf

from gammagl.data import HeteroGraph
from gammagl.models.dhn import DHNModel, NODE_TYPE, K_HOP, NUM_FEA, NUM_NEIGHBOR, type2idx
from gammagl.utils import k_hop_subgraph

# 全局忽略UserWarning
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

random.seed(0)  # 确保各个算法的数据，邻居等信息一致，保证公平

PATH = 'MA.txt'

G = HeteroGraph()


def load_ACM(test_ratio=0.2):
    edge_index_M = []
    edge_index_A = []

    with open(PATH, 'r') as f:
        for line in f.readlines():
            src, dst = line.strip().split()  # 假设每行包含两个由空格分隔的节点标识符
            src_type, src_id = src[0], src[1:]  # 解析源节点类型和ID
            dst_type, dst_id = dst[0], dst[1:]  # 解析目标节点类型和ID

            # 将节点ID转换为整数索引,并放入列表中
            if src[0] == 'M':
                edge_index_M.append(int(src_id))
            elif src[0] == 'A':
                edge_index_A.append(-int(src_id) - 1)

            if dst[0] == 'M':
                edge_index_M.append(int(dst_id))
            elif dst[0] == 'A':
                edge_index_A.append(-int(dst_id) - 1)

    edge_index = tlx.convert_to_tensor([edge_index_M, edge_index_A])
    G['M', 'MA', 'A'].edge_index = edge_index

    # 计算分割点
    sp = 1 - test_ratio * 2
    num_edge = len(edge_index_M)
    sp1 = int(num_edge * sp)
    sp2 = int(num_edge * test_ratio)

    G_train = HeteroGraph()
    G_val = HeteroGraph()
    G_test = HeteroGraph()

    # 划分训练集，验证集和测试集
    G_train['M', 'MA', 'A'].edge_index = tlx.convert_to_tensor([edge_index_M[:sp1], edge_index_A[:sp1]])
    G_val['M', 'MA', 'A'].edge_index = tlx.convert_to_tensor([edge_index_M[sp1:sp1 + sp2], edge_index_A[sp1:sp1 + sp2]])
    G_test['M', 'MA', 'A'].edge_index = tlx.convert_to_tensor([edge_index_M[sp1 + sp2:], edge_index_A[sp1 + sp2:]])

    print(
        f"all edge: {len(G['M', 'MA', 'A'].edge_index[0])}, train edge: {len(G_train['M', 'MA', 'A'].edge_index[0])}, val edge: {len(G_val['M', 'MA', 'A'].edge_index[0])}, test edge: {len(G_test['M', 'MA', 'A'].edge_index[0])}")

    return G_train, G_val, G_test


def find_all_simple_paths(edge_index, src, dest, max_length):
    # 将edge_index转换为邻接列表表示
    num_nodes = max(edge_index[0].max().item(),
                    edge_index[1].max().item(),
                    -edge_index[0].min().item(),
                    -edge_index[1].min().item(),
                    src.item()) + 1
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


def dist_encoder(src, dest, G, K_HOP, one_hot=True):
    if (G.size(1) == 0):
        paths = []
    else:
        paths = find_all_simple_paths(G, src, dest, K_HOP + 2)

    cnt = [K_HOP + 1] * NODE_TYPE  # 超过max_spd的默认截断
    for path in paths:
        res = [0] * NODE_TYPE
        for i in path:
            if i >= 0:
                res[type2idx['M']] += 1
            else:
                res[type2idx['A']] += 1

        for k in range(NODE_TYPE):
            cnt[k] = min(cnt[k], res[k])

    # 生成one-hot编码
    if one_hot:
        one_hot_list = [np.eye(K_HOP + 2, dtype=np.float64)[cnt[i]]
                        for i in range(NODE_TYPE)]
        return np.concatenate(one_hot_list)
    return cnt


def type_encoder(node):
    res = [0] * NODE_TYPE
    if node.item() >= 0:
        res[type2idx['M']] = 1.0
    else:
        res[type2idx['A']] = 1.0
    return res


mini_batch = []
fea_batch = []


def gen_fea_batch(G, root, fea_dict, hop):
    fea_batch = []
    mini_batch.append(root)

    a = [0] * (K_HOP + 2) * 4 + type_encoder(root)

    fea_batch.append(np.asarray(a,
                                dtype=np.float32
                                ).reshape(-1, NUM_FEA)
                     )

    # 一阶邻居采样
    ns_1 = []
    src, dst = G
    for node in mini_batch[-1]:
        if node.item() >= 0:
            neighbors_mask = src == node
        else:
            neighbors_mask = dst == node
        neighbors = list(dst[neighbors_mask].numpy())
        neighbors.append(node.item())
        random_choice_list = np.random.choice(neighbors, NUM_NEIGHBOR, replace=True)
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

    # 二阶邻居采样
    ns_2 = []
    for node in mini_batch[-1]:
        if node.item() >= 0:
            neighbors_mask = src == node
        else:
            neighbors_mask = dst == node
        neighbors = list(dst[neighbors_mask].numpy())
        neighbors.append(node.item())
        random_choice_list = np.random.choice(neighbors, NUM_NEIGHBOR, replace=True)
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


def subgraph_sampling_with_DE_node_pair(G, node_pair, K_HOP=2):
    [A, B] = node_pair

    edge_index = tlx.concat([G['M', 'MA', 'A'].edge_index, reversed(G['M', 'MA', 'A'].edge_index)], axis=1)

    # 求A和B的K跳子图
    sub_G_for_AB = k_hop_subgraph([A, B], K_HOP, edge_index)

    # 使用布尔索引删除边
    # 注意：只是删除边，点仍然保留
    edge_index_np = sub_G_for_AB[1].numpy()
    remove_indices = tlx.convert_to_tensor([
        ((edge_index_np[0, i] == A) & (edge_index_np[1, i] == B)) | (
                (edge_index_np[0, i] == B) & (edge_index_np[1, i] == A))
        for i in range(sub_G_for_AB[1].shape[1])
    ])
    remove_indices = remove_indices.numpy()
    sub_G_index = sub_G_for_AB[1][:, ~remove_indices]

    sub_G_nodes = set(np.unique(sub_G_for_AB[0].numpy())) | set(
        np.unique(sub_G_for_AB[1].numpy()))  # 获取图中的点
    sub_G_nodes = tlx.convert_to_tensor(list(sub_G_nodes))

    # 子图中所有点到node pair的距离
    SPD_based_on_node_pair = {}
    for node in sub_G_nodes:
        tmpA = dist_encoder(A, node, sub_G_index, K_HOP)
        tmpB = dist_encoder(B, node, sub_G_index, K_HOP)

        SPD_based_on_node_pair[node.item()] = np.concatenate([tmpA, tmpB], axis=0)

    A_fea_batch = gen_fea_batch(sub_G_index, A,
                                SPD_based_on_node_pair, K_HOP)
    B_fea_batch = gen_fea_batch(sub_G_index, B,
                                SPD_based_on_node_pair, K_HOP)

    return A_fea_batch, B_fea_batch


def batch_data(G,
               batch_size=3):
    edge_index = G['M', 'MA', 'A'].edge_index
    nodes = set(tlx.convert_to_tensor(np.unique(edge_index[0].numpy()))) | set(
        tlx.convert_to_tensor(np.unique(edge_index[1].numpy())))

    nodes_list = []
    for node in nodes:
        nodes_list.append(node.item())

    num_batch = int(len(edge_index[0]) / batch_size)

    # 打乱边的顺序
    edge_index_np = np.array(edge_index)
    permutation = np.random.permutation(edge_index_np.shape[1])  # 生成一个随机排列的索引
    edge_index_np = edge_index_np[:, permutation]  # 使用这个排列索引来打乱 edge_index
    edge_index = tlx.convert_to_tensor(edge_index_np)

    for idx in range(num_batch):
        batch_edge = edge_index[:, idx * batch_size:(idx + 1) * batch_size]  # 取出batch_size条边
        batch_label = [1.0] * batch_size

        batch_A_fea = []
        batch_B_fea = []
        batch_x = []
        batch_y = []

        i = 0
        for by in batch_label:
            bx = batch_edge[:, i:i + 1]

            # 正样本
            posA, posB = subgraph_sampling_with_DE_node_pair(G, bx, K_HOP=K_HOP)
            batch_A_fea.append(posA)
            batch_B_fea.append(posB)
            batch_y.append(np.asarray(by, dtype=np.float32))

            # 负样本
            neg_tmpB_id = random.choice(nodes_list)
            node_pair = tlx.convert_to_tensor([[bx[0].item()], [neg_tmpB_id]])

            negA, negB = subgraph_sampling_with_DE_node_pair(G, node_pair, K_HOP=K_HOP)
            batch_A_fea.append(negA)
            batch_B_fea.append(negB)
            batch_y.append(np.asarray(0.0, dtype=np.float32))

        yield np.asarray(np.squeeze(batch_A_fea)), np.asarray(np.squeeze(batch_B_fea)), np.asarray(
            batch_y).reshape(batch_size * 2, 1)


G_train, G_val, G_test = load_ACM(test_ratio=0.3)

m = DHNModel()

lr = 0.0002
optim = tlx.optimizers.Adam(lr=lr, weight_decay=0.01)

train_weights = m.trainable_weights


class Loss(tlx.model.WithLoss):
    def __init__(self, net, loss_fn):
        super(Loss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['n1'], data['n2'], data['label'])
        y = tlx.convert_to_tensor(y)
        loss = self._loss_fn(logits, y)
        return loss


net_with_loss = Loss(m, loss_fn=tlx.losses.sigmoid_cross_entropy)
net_with_train = TrainOneStep(net_with_loss, optim, train_weights)

EPOCH = 100
BATCH_SIZE = 32

tra_auc_cul = tf.keras.metrics.AUC()
val_auc_cul = tf.keras.metrics.AUC()
test_auc_cul = tf.keras.metrics.AUC()

for epoch in range(EPOCH):
    print("-----Epoch {}/{}-----".format(epoch + 1, EPOCH))

    # train
    m.set_train()
    tra_batch_A_fea, tra_batch_B_fea, tra_batch_y = batch_data(G_train, BATCH_SIZE).__next__()
    tra_out = m(tra_batch_A_fea, tra_batch_B_fea, tra_batch_y)

    data = {
        "n1": tra_batch_A_fea,
        "n2": tra_batch_B_fea,
        "label": tra_batch_y
    }

    tra_loss = net_with_train(data, tra_batch_y)
    tra_auc_cul.update_state(y_true=tra_batch_y, y_pred=tlx.sigmoid(tra_out).detach().numpy())
    tra_auc = tra_auc_cul.result().numpy()
    print('train: ', tra_loss, tra_auc)

    # val
    m.set_eval()
    val_batch_A_fea, val_batch_B_fea, val_batch_y = batch_data(G_val, BATCH_SIZE).__next__()
    val_out = m(val_batch_A_fea, val_batch_B_fea, val_batch_y)

    val_loss = tlx.losses.sigmoid_cross_entropy(output=val_out, target=tlx.convert_to_tensor(val_batch_y))
    val_auc_cul.update_state(y_true=val_batch_y, y_pred=tlx.sigmoid(val_out).detach().numpy())
    val_auc = val_auc_cul.result().numpy()
    print("val: ", val_loss.item(), val_auc)

    # test
    test_batch_A_fea, test_batch_B_fea, test_batch_y = batch_data(G_test, BATCH_SIZE).__next__()
    test_out = m(test_batch_A_fea, test_batch_B_fea, test_batch_y)

    test_loss = tlx.losses.sigmoid_cross_entropy(output=test_out, target=tlx.convert_to_tensor(test_batch_y))
    test_auc_cul.update_state(y_true=test_batch_y, y_pred=tlx.sigmoid(test_out).detach().numpy())
    test_auc = test_auc_cul.result().numpy()
    print("test: ", test_loss.item(), test_auc)
