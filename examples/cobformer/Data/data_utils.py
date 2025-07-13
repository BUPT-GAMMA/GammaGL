import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import metis
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import scatter


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]
    train_mask = torch.zeros_like(label, dtype=torch.bool)
    train_mask[train_idx] = True
    valid_mask = torch.zeros_like(label, dtype=torch.bool)
    valid_mask[valid_idx] = True
    test_mask = torch.zeros_like(label, dtype=torch.bool)
    test_mask[test_idx] = True

    return train_mask, valid_mask, test_mask


def load_fixed_splits(data_dir, dataset, name, protocol):
    splits_lst = []
    if name in ['Cora', 'CiteSeer', 'PubMed', 'ogbn-arxiv', 'ogbn-products'] and protocol == 'semi':
        splits = {}
        splits['train'] = torch.as_tensor(dataset.train_mask)
        splits['valid'] = torch.as_tensor(dataset.valid_mask)
        splits['test'] = torch.as_tensor(dataset.test_mask)
        splits['train'] = F.pad(splits['train'], [0, 1])
        splits['valid'] = F.pad(splits['valid'], [0, 1])
        splits['test'] = F.pad(splits['test'], [0, 1])
        splits_lst.append(splits)
    elif name in ['film', 'deezer']:
        for i in range(10):
            splits_file_path = '{}/{}'.format(data_dir, name) + '_split_50_25_' + str(i) + '.npz'
            splits = {}
            with np.load(splits_file_path) as splits_file:
                splits['train'] = torch.BoolTensor(splits_file['train_mask'])
                splits['valid'] = torch.BoolTensor(splits_file['val_mask'])
                splits['test'] = torch.BoolTensor(splits_file['test_mask'])
                splits['train'] = F.pad(splits['train'], [0, 1])
                splits['valid'] = F.pad(splits['valid'], [0, 1])
                splits['test'] = F.pad(splits['test'], [0, 1])
            splits_lst.append(splits)
    else:
        raise NotImplementedError

    return splits_lst


def class_rand_splits(label, label_num_per_class):
    train_idx, non_train_idx = [], []
    idx = torch.arange(label.shape[0])
    class_list = label.squeeze().unique()
    valid_num, test_num = 500, 1000
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:label_num_per_class].tolist()
        non_train_idx += rand_idx[label_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    non_train_idx = torch.as_tensor(non_train_idx)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]
    valid_idx, test_idx = non_train_idx[:valid_num], non_train_idx[valid_num:valid_num + test_num]
    train_mask = torch.zeros_like(label, dtype=torch.bool)
    train_mask[train_idx] = True
    valid_mask = torch.zeros_like(label, dtype=torch.bool)
    valid_mask[valid_idx] = True
    test_mask = torch.zeros_like(label, dtype=torch.bool)
    test_mask[test_idx] = True

    return train_mask, valid_mask, test_mask


def metis_partition(g, n_patches=50):
    if g['num_nodes'] < n_patches:
        membership = torch.randperm(n_patches)
    else:
        # data augmentation
        adjlist = g['edge_index'].t()
        G = nx.Graph()
        G.add_nodes_from(np.arange(g['num_nodes']))
        G.add_edges_from(adjlist.tolist())
        # metis partition
        cuts, membership = metis.part_graph(G, n_patches, recursive=True)

    assert len(membership) >= g['num_nodes']
    membership = torch.tensor(membership[:g['num_nodes']])


    patch = []
    max_patch_size = -1
    for i in range(n_patches):
        patch.append(list())
        patch[-1] = torch.where(membership == i)[0].tolist()
        max_patch_size = max(max_patch_size, len(patch[-1]))

    for i in range(len(patch)):
        l = len(patch[i])
        if l < max_patch_size:
            patch[i] += [g['num_nodes']] * (max_patch_size - l)

    patch = torch.tensor(patch)

    return patch


def patch2batch(g, node_mask):
    patches = node_mask.shape[0]
    max_patch_size = node_mask.sum(dim=1).max()
    all_nodes = torch.tensor(range(g['num_nodes']))
    batch_node_list = list()
    for i in range(patches):
        patch_nodes = all_nodes[node_mask[i, :]].tolist()
        l = len(patch_nodes)
        if l < max_patch_size:
            patch_nodes += [g['num_nodes']] * (max_patch_size - l)
        batch_node_list.append(patch_nodes)

    batch = torch.tensor(batch_node_list)
    return batch


def norm(edge_index, num_nodes=None, edge_weight=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    fill_value = 1.

    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=torch.float,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    # return torch.sparse_coo_tensor(edge_index, edge_weight)
    return edge_index, edge_weight
