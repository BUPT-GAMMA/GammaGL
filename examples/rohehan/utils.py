# -*- coding: UTF-8 -*-
import os
import numpy as np
import tensorlayerx as tlx
from sklearn.metrics import f1_score
from scipy import sparse, io as sio
from gammagl.data import HeteroGraph
import scipy.sparse as sp
from contextlib import nullcontext
from gammagl.data import download_url

# Evaluation function for accuracy and F1-score
def score(logits, labels):
    predictions = tlx.argmax(logits, axis=1)
    predictions = tlx.convert_to_numpy(predictions)
    labels = tlx.convert_to_numpy(labels)

    accuracy = np.sum(predictions == labels) / len(predictions)
    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    return accuracy, micro_f1, macro_f1

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

# Construct a heterogeneous graph with the given data
def get_hg(dataname, given_adj_dict, features_dict, labels=None, train_mask=None, val_mask=None, test_mask=None):
    meta_graph = HeteroGraph()
    meta_graph['paper'].x = features_dict['paper']
    meta_graph['paper'].num_nodes = features_dict['paper'].shape[0]

    # Add meta-path-based edges
    meta_graph['paper', 'author', 'paper'].edge_index = np.array(given_adj_dict['pa'].dot(given_adj_dict['ap']).nonzero())
    meta_graph['paper', 'field', 'paper'].edge_index = np.array(given_adj_dict['pf'].dot(given_adj_dict['fp']).nonzero())

    # Add labels and masks
    meta_graph['paper'].y = labels
    meta_graph['paper'].train_mask = train_mask
    meta_graph['paper'].val_mask = val_mask
    meta_graph['paper'].test_mask = test_mask

    return meta_graph

# Set random seed and device settings based on input arguments
def setup(args):
    tlx.set_seed(args.seed)
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")
    return args


# Compute the transition matrix for edge types based on meta-paths
def get_transition(given_hete_adjs, metapath_info, edge_index_dict, edge_types):
    hete_adj_dict_tmp = {}
    for key in given_hete_adjs.keys():
        deg = given_hete_adjs[key].sum(1).A1
        deg_inv = 1 / np.where(deg > 0, deg, 1)
        deg_inv_mat = sp.diags(deg_inv)
        hete_adj_dict_tmp[key] = deg_inv_mat.dot(given_hete_adjs[key])

    trans_edge_weights_list = []
    for i, metapath in enumerate(metapath_info):
        adj = hete_adj_dict_tmp[metapath[0][1]]
        for etype in metapath[1:]:
            adj = adj.dot(hete_adj_dict_tmp[etype[1]])

        edge_type = edge_types[i]
        edge_index = edge_index_dict[edge_type]

        edge_trans_values = adj[edge_index[0], edge_index[1]].A1
        trans_edge_weights_list.append(edge_trans_values)
    return trans_edge_weights_list

# Convert edge indices to a sparse adjacency matrix
def edge_index_to_adj_matrix(edge_index, num_src_nodes, num_dst_nodes):
    src, dst = edge_index
    data = np.ones(src.shape[0])
    adj_matrix = sp.csc_matrix((data, (src, dst)), shape=(num_src_nodes, num_dst_nodes))
    return adj_matrix

# Extract value from a tensor depending on the backend used
def to_item(tensor):
    if tlx.BACKEND == 'torch':
        return tensor.item()
    elif tlx.BACKEND == 'tensorflow':
        return tensor.numpy().item()
    else:
        raise NotImplementedError(f"Unsupported backend: {tlx.BACKEND}")

# Disable gradient computation
def no_grad():
    if tlx.BACKEND == 'torch':
        import torch
        return torch.no_grad()
    elif tlx.BACKEND == 'tensorflow':
        return nullcontext()
    elif tlx.BACKEND == 'paddle':
        import paddle
        return paddle.no_grad()
    elif tlx.BACKEND == 'mindspore':
        import mindspore
        return mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE)
    else:
        raise NotImplementedError(f"Unsupported backend: {tlx.BACKEND}")

def download_attack_data_files():
    """Download necessary ACM data files from GitHub if they are missing."""
    # Define the base URL for raw files in the repository
    base_url = "https://raw.githubusercontent.com/BUPT-GAMMA/RoHe/main/Code/data"

    # Files to download
    files_to_download = [
        "generated_attacks/adv_acm_pap_pa_1.pkl",
        "generated_attacks/adv_acm_pap_pa_3.pkl",
        "generated_attacks/adv_acm_pap_pa_5.pkl",
        "preprocess/target_nodes/acm_r_target0.pkl",
        "preprocess/target_nodes/acm_r_target1.pkl",
        "preprocess/target_nodes/acm_r_target2.pkl",
        "preprocess/target_nodes/acm_r_target3.pkl",
        "preprocess/target_nodes/acm_r_target4.pkl"
    ]

    # Download each file if it does not exist locally
    for file_path in files_to_download:
        # Construct the full URL and local save path
        file_url = f"{base_url}/{file_path}"
        save_folder = os.path.join("data", os.path.dirname(file_path))

        # Ensure directories exist and download the file
        if not os.path.exists(os.path.join(save_folder, os.path.basename(file_path))):
            download_url(file_url, save_folder)
