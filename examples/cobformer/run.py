from time import perf_counter as t

import torch
import torch.nn as nn
import random
from gammagl.models.CoBFormer import CoBFormer
from Train.train_test import *

# max_val = -10000
def co_early_stop_train(epochs, patience, model, data, label, patch, split_index, optimizer, show_details,
                        postfix, save_path=None):
    best_epoch1 = 0
    best_epoch2 = 0
    acc_val1_max = 0.
    acc_val2_max = 0.
    logger = []
    max_val = -10000

    for epoch in range(1, epochs + 1):

        micro_val1, micro_test1, macro_val1, macro_test1, micro_val2, micro_test2, macro_val2, macro_test2 = co_train(
            model, data, label, patch, split_index, optimizer)
        logger.append(
            [micro_val1, micro_test1, macro_val1, macro_test1, micro_val2, micro_test2, macro_val2, macro_test2])

        if show_details and epoch % 50 == 0:
            print(
                f'(T) | Epoch={epoch:03d}\n',
                f'micro_val1={micro_val1:.4f}, micro_test1={micro_test1:.4f}, macro_val1={macro_val1:.4f}, macro_test1={macro_test1:.4f}\n',
                f'micro_val2={micro_val2:.4f}, micro_test2={micro_test2:.4f}, macro_val2={macro_val2:.4f}, macro_test2={macro_test2:.4f}\n')

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

    return res_gnn, res_trans


def run(args, config, device, data, patch, split_idx, alpha, tau, postfix):
    learning_rate = args.learning_rate
    # learning_rate2 = args.learning_rate2

    weight_decay = args.weight_decay
    gcn_wd = args.gcn_wd
    num_hidden = config['num_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    num_layers = config['num_layers']
    n_head = config['n_head']
    num_epochs = config['num_epochs']
    gcn_type = args.gcn_type
    gcn_layers = args.gcn_layers
    gcn_use_bn = args.gcn_use_bn
    use_patch_attn = args.use_patch_attn
    show_details = args.show_details
    patch = patch.to(device)
    num_nodes = data.graph['num_nodes']
    num_classes = data.label.max() + 1
    num_features = data.graph['node_feat'].shape[-1]
    data.graph['node_feat'] = data.graph['node_feat'].to(device)
    data.graph['edge_index'] = data.graph['edge_index'].to(device)
    data.label = data.label.to(device)

    split_idx['train'] = split_idx['train'].to(device)
    split_idx['valid'] = split_idx['valid'].to(device)
    split_idx['test'] = split_idx['test'].to(device)

    label = F.one_hot(data.label, num_classes).float()

    # model = Beyondformer(num_nodes, num_features, num_hidden, num_classes, activation,
    #  layers=num_layers, gnn_layers=gcn_layers, n_head=n_head, alpha=alpha, ratio=ratio).to(device)
    model = CoBFormer(num_nodes, num_features, num_hidden, num_classes, activation, layers=num_layers,
                     gcn_layers=gcn_layers, gcn_type=gcn_type, n_head=n_head, alpha=alpha, tau=tau,
                     gcn_use_bn=gcn_use_bn, use_patch_attn=use_patch_attn).to(device)
    # print(model)

    if args.dataset in ['film', 'CiteSeer', 'Cora', 'PubMed', "Deezer"]:
        optimizer = torch.optim.Adam([
            {'params': model.bga.parameters(), 'weight_decay': weight_decay},
            {'params': model.gcn.parameters(), 'weight_decay': gcn_wd}
        ], lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    patience = num_epochs

    res_gnn, res_trans = co_early_stop_train(
        num_epochs, patience,
        model, data, label,
        patch, split_idx,
        optimizer, show_details,
        postfix)
    print("=== Train Final ===")
    print(
        f'micro_val1={res_gnn[0]:.4f}, micro_test1={res_gnn[1]:.4f}, macro_val1={res_gnn[2]:.4f}, macro_test1={res_gnn[3]:.4f}, micro_best1={res_gnn[4]:.4f}, macro_best1={res_gnn[5]:.4f},\n',
        f'micro_val2={res_trans[0]:.4f}, micro_test2={res_trans[1]:.4f}, macro_val2={res_trans[2]:.4f}, macro_test2={res_trans[3]:.4f}, micro_best2={res_trans[4]:.4f}, macro_best2={res_trans[5]:.4f}\n')

    return res_gnn, res_trans
