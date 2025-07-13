from time import perf_counter as t

import torch
import random
from gammagl.models.CoBFormer import *
from torch_geometric.utils import subgraph
from torch_geometric.utils.map import map_index
from Train.train_test import *
import torch.nn as nn


def co_early_stop_train_batch(epochs, patience, model, data, label, patch, batch_size, split_index, optimizer,
                              show_details, device, postfix):
    best_epoch1 = 0
    best_epoch2 = 0
    acc_val1_max = 0.
    acc_val2_max = 0.
    logger = []

    n_patch, patch_size = patch.shape
    patch_per_batch = batch_size // patch_size
    num_batch = n_patch // patch_per_batch + (n_patch % patch_per_batch > 0)

    for epoch in range(1, epochs + 1):

        idx = torch.randperm(n_patch)
        for i in range(num_batch):
            patch_idx = idx[i * patch_per_batch: (i + 1) * patch_per_batch]
            patch_i = patch[patch_idx]
            node_i = torch.unique(patch_i)
            patch_i = map_index(patch_i, node_i)[0].view(patch_i.shape).to(device)
            node_feat_i = data.graph['node_feat'][node_i].to(device)
            edge_index_i, _ = subgraph(node_i, data.graph['edge_index'], num_nodes=data.graph['num_nodes'],
                                       relabel_nodes=True)
            edge_index_i = edge_index_i.to(device)
            label_i = label[node_i].to(device)
            train_idx = split_index['train'][node_i].to(device)

            co_train_batch(model, node_feat_i, edge_index_i, label_i, patch_i, train_idx, optimizer)

        if epoch % 5 == 0:
            model.eval()
            y1 = torch.zeros_like(data.label).to(device)
            y2 = torch.zeros_like(data.label).to(device)
            with torch.no_grad():
                idx = torch.randperm(n_patch)
                for i in range(num_batch):
                    patch_idx = idx[i * patch_per_batch: (i + 1) * patch_per_batch]
                    patch_i = patch[patch_idx]
                    node_i = torch.unique(patch_i)
                    patch_i = map_index(patch_i, node_i)[0].view(patch_i.shape).to(device)
                    node_feat_i = data.graph['node_feat'][node_i].to(device)
                    edge_index_i, _ = subgraph(node_i, data.graph['edge_index'], num_nodes=data.graph['num_nodes'],
                                               relabel_nodes=True)
                    edge_index_i = edge_index_i.to(device)
                    pred1, pred2 = model(node_feat_i, patch_i, edge_index_i)
                    pred1 = torch.argmax(pred1, dim=1).squeeze()
                    pred2 = torch.argmax(pred2, dim=1).squeeze()
                    y1[node_i] = pred1
                    y2[node_i] = pred2
                y = torch.tensor(data.label).to(device)
                num_classes = data.label.max() + 1
                micro_val1, macro_val1 = eval_f1(y1[split_index['valid']], y[split_index['valid']], num_classes)
                micro_test1, macro_test1 = eval_f1(y1[split_index['test']], y[split_index['test']], num_classes)
                micro_val2, macro_val2 = eval_f1(y2[split_index['valid']], y[split_index['valid']], num_classes)
                micro_test2, macro_test2 = eval_f1(y2[split_index['test']], y[split_index['test']], num_classes)
                acc1 = torch.eq(y1[split_index['test']], y[split_index['test']]).float().mean()
                acc2 = torch.eq(y2[split_index['test']], y[split_index['test']]).float().mean()


                logger.append(
                    [micro_val1, micro_test1, macro_val1, macro_test1, micro_val2, micro_test2, macro_val2, macro_test2])

            if show_details and epoch % 5 == 0:
                print(
                    f'(T) | Epoch={epoch:03d}\n',
                    f'micro_val1={micro_val1:.4f}, micro_test1={micro_test1:.4f}, acc1={acc1:.4f}, macro_val1={macro_val1:.4f}, macro_test1={macro_test1:.4f}\n',
                    f'micro_val2={micro_val2:.4f}, micro_test2={micro_test2:.4f}, acc2={acc2:.4f}, macro_val2={macro_val2:.4f}, macro_test2={macro_test2:.4f}\n')
        # acc_val = (acc_val1 + acc_val2) /2.
        # if acc_val > acc_val1_max:
        #     acc_val1_max = acc_val
        #     best_epoch1 = epoch
        #     torch.save(model.state_dict(), f"tem/weight_best_pretrain_{postfix}_1.pkl")
        # if acc_val2 > acc_val2_max:
        #     acc_val2_max = acc_val2
        #     best_epoch2 = epoch
        #     torch.save(model.state_dict(), f"tem/weight_best_pretrain_{postfix}_2.pkl")

    logger = torch.tensor(logger)
    ind = torch.argmax(logger, dim=0)

    res_gnn = []
    res_trans = []

    res_gnn.append(logger[ind[0]][0])
    res_gnn.append(logger[ind[0]][1])
    res_gnn.append(logger[ind[2]][2])
    res_gnn.append(logger[ind[2]][3])
    res_gnn.append(logger[ind[1]][1])
    res_gnn.append(logger[ind[3]][3])

    res_trans.append(logger[ind[4]][4])
    res_trans.append(logger[ind[4]][5])
    res_trans.append(logger[ind[6]][6])
    res_trans.append(logger[ind[6]][7])
    res_trans.append(logger[ind[5]][5])
    res_trans.append(logger[ind[7]][7])

    # model.load_state_dict(torch.load(f"tem/weight_best_pretrain_{postfix}_1.pkl"))
    # acc_val1, acc_test1, acc_val2, acc_test2 = co_test(model, data, patch, split_index)
    # model.load_state_dict(torch.load(f"tem/weight_best_pretrain_{postfix}_2.pkl"))
    # _, _, acc_val2, acc_test2 = co_test(model, data, patch, split_index)
    return res_gnn, res_trans


def run_batch(args, config, device, data, patch, batch_size, split_idx, alpha, tau, postfix):
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_hidden = config['num_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    num_layers = 1
    n_head = config['n_head']
    num_epochs = config['num_epochs']
    gcn_type = args.gcn_type
    gcn_layers = args.gcn_layers
    gcn_use_bn = args.gcn_use_bn
    show_details = args.show_details
    patch = patch
    num_nodes = data.graph['num_nodes']
    num_classes = data.label.max() + 1
    num_features = data.graph['node_feat'].shape[-1]

    label = F.one_hot(data.label, num_classes).float()

    # model = Beyondformer(num_nodes, num_features, num_hidden, num_classes, activation,
    #  layers=num_layers, gnn_layers=gcn_layers, n_head=n_head, alpha=alpha, ratio=ratio).to(device)
    model = CoBFormer(num_nodes, num_features, num_hidden, num_classes, activation, layers=num_layers,
                     gcn_layers=gcn_layers, gcn_type=gcn_type, n_head=n_head, alpha=alpha, tau=tau,
                     gcn_use_bn=gcn_use_bn).to(device)
    # print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    patience = num_epochs

    # best_epoch, acc_val, acc_test = early_stop_train(num_epochs, patience, model, data, label, patch, patch_adj,
    #                                                  optimizer, show_details, postfix)
    # print("=== Train Final ===")
    # print(
    #     f"best_epoch: {best_epoch}, acc_val: {acc_val}, acc_test: {acc_test}")

    res_gnn, res_trans = co_early_stop_train_batch(
        num_epochs, patience,
        model, data, label,
        patch, batch_size, split_idx,
        optimizer, show_details,
        device, postfix)
    print("=== Train Final ===")
    print(
        f'micro_val1={res_gnn[0]:.4f}, micro_test1={res_gnn[1]:.4f}, macro_val1={res_gnn[2]:.4f}, macro_test1={res_gnn[3]:.4f}, micro_best1={res_gnn[4]:.4f}, macro_best1={res_gnn[5]:.4f},\n',
        f'micro_val2={res_trans[0]:.4f}, micro_test2={res_trans[1]:.4f}, macro_val2={res_trans[2]:.4f}, macro_test2={res_trans[3]:.4f}, micro_best2={res_trans[4]:.4f}, macro_best2={res_trans[5]:.4f}\n')

    return res_gnn, res_trans
