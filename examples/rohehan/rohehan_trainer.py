# -*- coding: UTF-8 -*-
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR
import argparse
import numpy as np
import tensorlayerx as tlx
from gammagl.models import RoheHAN
from utils import *
import pickle as pkl
from gammagl.utils.convert import edge_index_to_adj_matrix
from gammagl.datasets.acm4rohe import ACM4Rohe

class SemiSpvzLoss(tlx.nn.Module):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__()
        self.net = net
        self.loss_fn = loss_fn

    def forward(self, data, y):
        logits = self.net(data['x_dict'], data['edge_index_dict'], data['num_nodes_dict'])
        train_logits = tlx.gather(logits['paper'], data['train_idx'])
        train_y = tlx.gather(y, data['train_idx'])
        loss = self.loss_fn(train_logits, train_y)
        return loss

# Evaluate the model, returning loss and accuracy scores
def evaluate(model, data, labels, mask, loss_func):
    model.set_eval()
    logits = model(data['x_dict'], data['edge_index_dict'], data['num_nodes_dict'])
    logits = logits['paper']  # Focus evaluation on 'paper' nodes
    mask_indices = mask  # Assuming mask is an array of indices
    logits_masked = tlx.gather(logits, tlx.convert_to_tensor(mask_indices, dtype=tlx.int64))
    labels_masked = tlx.gather(labels, tlx.convert_to_tensor(mask_indices, dtype=tlx.int64))
    loss = loss_func(logits_masked, labels_masked)

    accuracy, micro_f1, macro_f1 = score(logits_masked, labels_masked)
    return loss, accuracy, micro_f1, macro_f1

def main(args):
    # Load ACM raw dataset
    dataname = 'acm'
    dataset = ACM4Rohe(root = "./")
    g = dataset[0]
    dataset.download_attack_data_files()
    features_dict = {ntype: g[ntype].x for ntype in g.node_types if hasattr(g[ntype], 'x')}
    labels = g['paper'].y
    train_mask = g['paper'].train_mask
    val_mask = g['paper'].val_mask
    test_mask = g['paper'].test_mask

    # Compute number of classes
    num_classes = int(tlx.reduce_max(labels)) + 1

    # Get train_idx, val_idx, test_idx from masks
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    x_dict = features_dict
    y = labels
    features = features_dict['paper']

    # Define meta-paths (PAP, PFP)
    meta_paths = [[('paper', 'pa', 'author'), ('author', 'ap', 'paper')],
                  [('paper', 'pf', 'field'), ('field', 'fp', 'paper')]]

    # Define initial settings for each edge type
    settings = {
        ('paper', 'author', 'paper'): {'T': 3, 'TransM': None},
        ('paper', 'field', 'paper'): {'T': 5, 'TransM': None},
    }

    # Prepare adjacency matrices
    hete_adjs = {
        'pa': edge_index_to_adj_matrix(g['paper', 'pa', 'author'].edge_index, g['paper'].num_nodes, g['author'].num_nodes),
        'ap': edge_index_to_adj_matrix(g['author', 'ap', 'paper'].edge_index, g['author'].num_nodes, g['paper'].num_nodes),
        'pf': edge_index_to_adj_matrix(g['paper', 'pf', 'field'].edge_index, g['paper'].num_nodes, g['field'].num_nodes),
        'fp': edge_index_to_adj_matrix(g['field', 'fp', 'paper'].edge_index, g['field'].num_nodes, g['paper'].num_nodes)
    }
    meta_g = dataset.get_meta_graph(dataname, hete_adjs, features_dict, labels, train_mask, val_mask, test_mask)
    # Prepare edge index and node count dictionaries
    edge_index_dict = {etype: meta_g[etype].edge_index for etype in meta_g.edge_types}
    num_nodes_dict = {ntype: meta_g[ntype].num_nodes for ntype in meta_g.node_types}

    # Compute edge transformation matrices
    trans_edge_weights_list = get_transition(hete_adjs, meta_paths, edge_index_dict, meta_g.metadata()[1])
    for i, edge_type in enumerate(meta_g.metadata()[1]):
        settings[edge_type]['TransM'] = trans_edge_weights_list[i]

    layer_settings = [settings, settings]

    # Initialize the RoheHAN model
    model = RoheHAN(
        metadata=meta_g.metadata(),
        in_channels=features.shape[1],
        hidden_size=args.hidden_units,
        out_size=num_classes,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
        settings=layer_settings
    )

    # Define optimizer and loss function
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
    loss_func = tlx.losses.softmax_cross_entropy_with_logits
    semi_spvz_loss = SemiSpvzLoss(model, loss_func)

    # Prepare training components
    train_weights = model.trainable_weights
    train_one_step = tlx.model.TrainOneStep(semi_spvz_loss, optimizer, train_weights)

    # Prepare data dictionary
    data = {
        "x_dict": x_dict,
        "edge_index_dict": edge_index_dict,
        "num_nodes_dict": num_nodes_dict,
        "train_idx": tlx.convert_to_tensor(train_idx, dtype=tlx.int64),
        "val_idx": tlx.convert_to_tensor(val_idx, dtype=tlx.int64),
        "test_idx": tlx.convert_to_tensor(test_idx, dtype=tlx.int64),
        "y": y
    }

    # Ensure the best model path exists
    if not os.path.exists(args.best_model_path):
        os.makedirs(args.best_model_path)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(args.num_epochs):
        model.set_train()
        # Forward and backward pass
        loss = train_one_step(data, y)

        # Evaluate on validation set
        model.set_eval()
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, data, y, val_idx, loss_func)

        print(f"Epoch {epoch+1} | Train Loss: {loss.item():.4f} | Val Micro-F1: {val_micro_f1:.4f} | Val Macro-F1: {val_macro_f1:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save model weights
            model.save_weights(os.path.join(args.best_model_path, 'best_model.npz'), format='npz_dict')

    # Load the best model
    model.load_weights(os.path.join(args.best_model_path, 'best_model.npz'), format='npz_dict')

    # Test the model
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, data, y, test_idx, loss_func)
    print(f"Test Micro-F1: {test_micro_f1:.4f} | Test Macro-F1: {test_macro_f1:.4f}")

    # Load target node IDs
    print("Loading target nodes")
    tar_idx = []
    # can attack 500 target nodes by seting range(5)
    for i in range(1):
        with open(f'data/preprocess/target_nodes/{dataname}_r_target{i}.pkl', 'rb') as f:
            tar_tmp = np.sort(pkl.load(f))
        tar_idx.extend(tar_tmp)

    # Evaluate on target nodes
    model.set_eval()
    logits_dict = model(data['x_dict'], data['edge_index_dict'], data['num_nodes_dict'])
    logits_clean = tlx.gather(logits_dict['paper'], tlx.convert_to_tensor(tar_idx, dtype=tlx.int64))
    labels_clean = tlx.gather(y, tlx.convert_to_tensor(tar_idx, dtype=tlx.int64))
    _, tar_micro_f1_clean, tar_macro_f1_clean = score(logits_clean, labels_clean)
    print(f"Clean data: Micro-F1: {tar_micro_f1_clean:.4f} | Macro-F1: {tar_macro_f1_clean:.4f}")

    # Load adversarial attacks
    n_perturbation = 1
    adv_filename = f'data/generated_attacks/adv_acm_pap_pa_{n_perturbation}.pkl'
    with open(adv_filename, 'rb') as f:
        modified_opt = pkl.load(f)

    # Apply adversarial attack
    logits_adv_list = []
    labels_adv_list = []
    for items in modified_opt:
        target_node = items[0]
        del_list = items[2]
        add_list = items[3]
        if target_node not in tar_idx:
            continue

        # Modify adjacency matrices for the attack
        mod_hete_adj_dict = {}
        for key in hete_adjs.keys():
            mod_hete_adj_dict[key] = hete_adjs[key].tolil()

        # Delete and add edges
        for edge in del_list:
            mod_hete_adj_dict['pa'][edge[0], edge[1]] = 0
            mod_hete_adj_dict['ap'][edge[1], edge[0]] = 0
        for edge in add_list:
            mod_hete_adj_dict['pa'][edge[0], edge[1]] = 1
            mod_hete_adj_dict['ap'][edge[1], edge[0]] = 1

        for key in mod_hete_adj_dict.keys():
            mod_hete_adj_dict[key] = mod_hete_adj_dict[key].tocsc()

        # Update edge index dictionary for the attack
        edge_index_dict_atk = {}
        meta_path_atk = [('paper', 'author', 'paper'), ('paper', 'field', 'paper')]
        for idx, edge_type in enumerate(meta_path_atk):
            # Recompute adjacency matrices for the attack
            if edge_type == ('paper', 'author', 'paper'):
                adj_matrix = mod_hete_adj_dict['pa'].dot(mod_hete_adj_dict['ap'])
            elif edge_type == ('paper', 'field', 'paper'):
                adj_matrix = mod_hete_adj_dict['pf'].dot(mod_hete_adj_dict['fp'])
            else:
                raise KeyError(f"Unknown edge type {edge_type}")

            src, dst = adj_matrix.nonzero()
            edge_index = np.vstack((src, dst))
            edge_index_dict_atk[edge_type] = edge_index

        # Update transformation matrices for the attack
        trans_edge_weights_list = get_transition(mod_hete_adj_dict, meta_paths, edge_index_dict_atk, meta_path_atk)

        for i, edge_type in enumerate(meta_path_atk):
            key = '__'.join(edge_type)
            if key in model.layer_list[0].gat_layers:
                model.layer_list[0].gat_layers[key].settings['TransM'] = trans_edge_weights_list[i]
            else:
                raise KeyError(f"Edge type key '{key}' not found in gat_layers.")

        # Prepare modified graph and data
        mod_features_dict = {'paper': features}
        g_atk = dataset.get_meta_graph(dataname, mod_hete_adj_dict, mod_features_dict, y, train_mask, val_mask, test_mask)
        data_atk = {
            "x_dict": g_atk.x_dict,
            "edge_index_dict": {etype: g_atk[etype].edge_index for etype in g_atk.edge_types},
            "num_nodes_dict": {ntype: g_atk[ntype].num_nodes for ntype in g_atk.node_types},
        }

        # Run the model on the attacked graph
        model.set_eval()
        with no_grad():
            logits_dict_atk = model(data_atk['x_dict'], data_atk['edge_index_dict'], data_atk['num_nodes_dict'])
            logits_atk = logits_dict_atk['paper']
            logits_adv = tlx.gather(logits_atk, tlx.convert_to_tensor([target_node], dtype=tlx.int64))
            label_adv = tlx.gather(y, tlx.convert_to_tensor([target_node], dtype=tlx.int64))

        logits_adv_list.append(logits_adv)
        labels_adv_list.append(label_adv)

    logits_adv = tlx.concat(logits_adv_list, axis=0)
    labels_adv = tlx.concat(labels_adv_list, axis=0)

    # Evaluate adversarial attack
    _, tar_micro_f1_atk, tar_macro_f1_atk = score(logits_adv, labels_adv)
    print(f"Attacked data: Micro-F1: {tar_micro_f1_atk:.4f} | Macro-F1: {tar_macro_f1_atk:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2, help="Random seed.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--num_heads", type=int, default=[8], help="Number of attention heads.")
    parser.add_argument("--hidden_units", type=int, default=8, help="Hidden units.")
    parser.add_argument("--dropout", type=float, default=0.6, help="Dropout rate.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index. Use -1 for CPU.")
    parser.add_argument("--best_model_path", type=str, default='./', help="Path to save the best model.")
    args = parser.parse_args()

    # Setup configuration
    tlx.set_seed(args.seed)
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    main(args)
