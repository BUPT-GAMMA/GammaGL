# -*- coding: UTF-8 -*-
import os
import numpy as np
import tensorlayerx as tlx
from sklearn.metrics import f1_score 
import scipy.sparse as sp
from contextlib import nullcontext

# Evaluation function for accuracy and F1-score
def score(logits, labels):
    predictions = tlx.argmax(logits, axis=1)
    predictions = tlx.convert_to_numpy(predictions)
    labels = tlx.convert_to_numpy(labels)

    accuracy = np.sum(predictions == labels) / len(predictions)
    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    return accuracy, micro_f1, macro_f1

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