import os
os.environ['TL_BACKEND'] = 'torch' 

import argparse
import numpy as np
import tensorlayerx as tlx
from gammagl.models.han_rohe import HAN
from examples.han_rohe.utils import *
import pickle as pkl
import copy

class WithLoss(tlx.nn.Module):
    def __init__(self, net, loss_fn):
        super(WithLoss, self).__init__()
        self.net = net
        self.loss_fn = loss_fn

    def forward(self, data, y):
        logits = self.net(data['x_dict'], data['edge_index_dict'], data['num_nodes_dict'])
        train_logits = tlx.gather(logits['paper'], data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self.loss_fn(train_logits, train_y)
        return loss

class TrainOneStep(tlx.nn.Module):
    def __init__(self, model, optimizer, train_weights):
        super(TrainOneStep, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_weights = train_weights

    def forward(self, data, labels):
        # 前向传播计算损失
        loss = self.model(data, labels)

        # 计算梯度并更新权重
        grads = self.optimizer.gradient(loss, self.train_weights)
        self.optimizer.apply_gradients(grads_and_vars=grads)

        return loss


def get_transition(given_hete_adjs, metapath_info):
    """
    生成基于元路径的转换矩阵。
    
    Parameters
    ----------
    given_hete_adjs : dict
        异构图中的邻接矩阵字典。
    metapath_info : list
        元路径的信息列表。
    
    Returns
    -------
    homo_adj_list : list
        同质图的转换矩阵列表。
    """
    hete_adj_dict_tmp = {}
    for key in given_hete_adjs.keys():
        deg = given_hete_adjs[key].sum(1).A1
        deg_inv = 1 / np.where(deg > 0, deg, 1)
        deg_inv_mat = sp.diags(deg_inv)
        hete_adj_dict_tmp[key] = deg_inv_mat.dot(given_hete_adjs[key])

    homo_adj_list = []
    for metapath in metapath_info:
        adj = hete_adj_dict_tmp[metapath[0][1]]
        for etype in metapath[1:]:
            adj = adj.dot(hete_adj_dict_tmp[etype[1]])
        homo_adj_list.append(adj)
    return homo_adj_list




def main(args):
    # 设置随机种子和设备
    set_random_seed(args.seed)
    device = tlx.set_device("CPU")

    # 加载数据
    dataname = 'acm'
    g, meta_g, features_dict, labels, num_classes, train_idx, val_idx, test_idx, \
    train_mask, val_mask, test_mask = load_acm_raw()

    x_dict = features_dict
    y = labels
    features = features_dict['paper']

    # 定义元路径
    meta_paths = [[('paper', 'pa', 'author'), ('author', 'ap', 'paper')],
                               [('paper', 'pf', 'field'), ('field', 'fp', 'paper')]]

    # 定义 settings，使用元组作为键
    settings = {
        ('paper', 'author', 'paper'): {'T': 2, 'TransM': None},
        ('paper', 'field', 'paper'): {'T': 5, 'TransM': None},
    }

    # 准备 hete_adjs
    def edge_index_to_adj_matrix(edge_index, num_src_nodes, num_dst_nodes):
        """将 edge_index 转换为 SciPy 的稀疏邻接矩阵"""
        src, dst = edge_index
        data = np.ones(src.shape[0])
        adj_matrix = sp.csc_matrix((data, (src, dst)), shape=(num_src_nodes, num_dst_nodes))
        return adj_matrix

    hete_adjs = {
        'pa': edge_index_to_adj_matrix(g['paper', 'pa', 'author'].edge_index, g['paper'].num_nodes, g['author'].num_nodes),
        'ap': edge_index_to_adj_matrix(g['author', 'ap', 'paper'].edge_index, g['author'].num_nodes, g['paper'].num_nodes),
        'pf': edge_index_to_adj_matrix(g['paper', 'pf', 'field'].edge_index, g['paper'].num_nodes, g['field'].num_nodes),
        'fp': edge_index_to_adj_matrix(g['field', 'fp', 'paper'].edge_index, g['field'].num_nodes, g['paper'].num_nodes)
    }

    # 生成转换矩阵并将其分配到正确的元路径
    trans_adj_list = get_transition(hete_adjs, meta_paths)
    print(meta_g.metadata())
    for i, edge_type in enumerate(meta_g.metadata()[1]):
        settings[edge_type]['TransM'] = trans_adj_list[i]

    layer_settings = [settings, settings]        
    
    # 准备 edge_index_dict 和 num_nodes_dict
    edge_index_dict = {}
    for etype in meta_g.edge_types:
        edge_index = meta_g[etype].edge_index
        edge_index_dict[etype] = edge_index

    num_nodes_dict = {ntype: meta_g[ntype].num_nodes for ntype in meta_g.node_types}


    # 定义模型
    model = HAN(
        metadata=meta_g.metadata(),
        in_channels=features.shape[1],
        hidden_size=args.hidden_units,
        out_size=num_classes,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
        settings=layer_settings  # 这里传递的是使用元组键的 settings 字典
    )

    # 定义优化器和损失函数
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
    loss_func = tlx.losses.softmax_cross_entropy_with_logits

    # 准备训练组件
    train_weights = model.trainable_weights
    train_one_step = TrainOneStep(WithLoss(model, loss_func), optimizer, train_weights)

    # 准备数据字典
    data = {
        "x_dict": x_dict,
        "edge_index_dict": edge_index_dict,
        "num_nodes_dict": num_nodes_dict,
        "train_idx": tlx.convert_to_tensor(train_idx, dtype=tlx.int64),
        "val_idx": tlx.convert_to_tensor(val_idx, dtype=tlx.int64),
        "test_idx": tlx.convert_to_tensor(test_idx, dtype=tlx.int64),
        "y": y
    }

    # 训练循环
    stopper = EarlyStopping(patience=args.patience)

    for epoch in range(args.num_epochs):
        model.set_train()
        # 前向和反向传播
        loss = train_one_step(data, y)
        print(f"Epoch {epoch} | Train Loss: {loss.item()}")

        # 在验证集上评估
        model.set_eval()
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, data, y, val_idx, loss_func)
        early_stop = stopper.step(val_loss.item(), val_acc, model)
        print(f"Epoch {epoch} | Val Micro-F1: {val_micro_f1:.4f} | Val Macro-F1: {val_macro_f1:.4f}")
        if early_stop:
            print("Early stopping")
            break

    # 加载最佳模型
    stopper.load_checkpoint(model)

    # 测试模型
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, data, y, test_idx, loss_func)
    print(f"Test Micro-F1: {test_micro_f1:.4f} | Test Macro-F1: {test_macro_f1:.4f}")

    # 加载目标节点 ID
    tar_idx = []
    for i in range(1):  # 可以通过更改范围攻击更多目标节点
        with open(f'data/preprocess/target_nodes/{dataname}_r_target{i}.pkl', 'rb') as f:
            tar_tmp = np.sort(pkl.load(f))
        tar_idx.extend(tar_tmp)

    # 在目标节点上评估
    model.set_eval()
    logits_dict = model(data['x_dict'], data['edge_index_dict'], data['num_nodes_dict'])
    logits_clean = tlx.gather(logits_dict['paper'], tlx.convert_to_tensor(tar_idx, dtype=tlx.int64))
    labels_clean = tlx.gather(y, tlx.convert_to_tensor(tar_idx, dtype=tlx.int64))
    _, tar_micro_f1_clean, tar_macro_f1_clean = score(logits_clean, labels_clean)
    print(f"Clean data: Micro-F1: {tar_micro_f1_clean:.4f} | Macro-F1: {tar_macro_f1_clean:.4f}")

    # 加载对抗攻击
    n_perturbation = 1
    adv_filename = f'data/generated_attacks/adv_acm_pap_pa_{n_perturbation}.pkl'
    with open(adv_filename, 'rb') as f:
        modified_opt = pkl.load(f)

    # 应用对抗攻击
    logits_adv_list = []
    labels_adv_list = []
    for items in modified_opt:
        target_node = items[0]
        del_list = items[2]
        add_list = items[3]
        if target_node not in tar_idx:
            continue

        # 修改邻接矩阵
        mod_hete_adj_dict = copy.deepcopy(hete_adjs)
        for edge in del_list:
            mod_hete_adj_dict['pa'][edge[0], edge[1]] = 0
            mod_hete_adj_dict['ap'][edge[1], edge[0]] = 0
        for edge in add_list:
            mod_hete_adj_dict['pa'][edge[0], edge[1]] = 1
            mod_hete_adj_dict['ap'][edge[1], edge[0]] = 1

        # 更新转换矩阵
        trans_adj_list = get_transition(mod_hete_adj_dict, meta_paths)
        for i in range(len(trans_adj_list)):
            model.layers[0].gat_layers[i].settings['TransM'] = trans_adj_list[i]

        # 重构图
        mod_features_dict = {'paper': features}
        g_atk = get_hg(dataname, mod_hete_adj_dict, mod_features_dict)

        # 准备被攻击的图的数据字典
        data_atk = {
            "x_dict": g_atk.x_dict,
            "edge_index_dict": g_atk.edge_index_dict,
            "num_nodes_dict": g_atk.num_nodes_dict,
        }

        # 在被攻击的图上运行模型
        model.set_eval()
        logits_dict_atk = model(data_atk['x_dict'], data_atk['edge_index_dict'], data_atk['num_nodes_dict'])
        logits_atk = logits_dict_atk['paper']
        logits_adv = tlx.gather(logits_atk, tlx.convert_to_tensor([target_node], dtype=tlx.int64))
        label_adv = tlx.gather(y, tlx.convert_to_tensor([target_node], dtype=tlx.int64))

        logits_adv_list.append(logits_adv)
        labels_adv_list.append(label_adv)

    logits_adv = tlx.concat(logits_adv_list, axis=0)
    labels_adv = tlx.concat(labels_adv_list, axis=0)

    # 评估对抗攻击
    _, tar_micro_f1_atk, tar_macro_f1_atk = score(logits_adv, labels_adv)
    print(f"Attacked data: Micro-F1: {tar_micro_f1_atk:.4f} | Macro-F1: {tar_macro_f1_atk:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2, help="Random seed.")
    parser.add_argument("--hetero", action='store_true', help="Use heterogeneous graph.")
    parser.add_argument("--log_dir", type=str, default='results', help="Log directory.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--hidden_units", type=int, default=8, help="Hidden units.")
    parser.add_argument("--dropout", type=float, default=0.6, help="Dropout rate.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay.")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=100, help="Patience for early stopping.")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index. Use -1 for CPU.")
    args = parser.parse_args()

    # 设置配置
    args = setup(args)

    main(args)
