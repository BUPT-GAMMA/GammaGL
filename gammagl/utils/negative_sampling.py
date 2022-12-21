import random

import numpy as np
import tensorlayerx as tlx

from gammagl.utils import coalesce, degree, remove_self_loops
from .num_nodes import maybe_num_nodes

def negative_sampling(edge_index, num_nodes = None, num_neg_samples = None, method = 'sparse', force_undirected = False):
    assert method in ['sparse', 'dense']

    bipartite = isinstance(num_nodes, (tuple, list))
    if num_nodes is None:
        num_nodes = maybe_num_nodes(edge_index)
    if bipartite:
        force_undirected = False
    else:
        num_nodes = (num_nodes, num_nodes)

    idx, population = edge_index_to_vector(edge_index, num_nodes, bipartite, force_undirected)

    if tlx.convert_to_numpy(idx).size >= population:
        return tlx.convert_to_tensor([[],[]], dtype=edge_index.dtype)
    
    if num_neg_samples is None:
        num_neg_samples = edge_index.shape[1]
    if force_undirected:
        num_neg_samples = num_neg_samples // 2
    
    prob = 1. - tlx.convert_to_numpy(idx).size / population
    sample_size = int(1.1 * num_neg_samples / prob)

    neg_idx = None
    if method == 'dense':
        mask = tlx.ones((population,), dtype=tlx.bool)
        if tlx.BACKEND == 'paddle':
            mask[idx] = False
        else:
            mask = tlx.scatter_update(mask, idx, tlx.zeros((idx.shape[0],), dtype=tlx.bool))
        for _ in range(3):
            rnd = sample(population, sample_size)
            rnd = tlx.mask_select(rnd, tlx.gather(mask, rnd))
            if neg_idx is None:
                neg_idx = rnd
            else:
                neg_idx = tlx.concat([neg_idx, rnd], axis=0)
            if tlx.convert_to_numpy(neg_idx).size >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break
            if tlx.BACKEND == 'paddle':
                mask[neg_idx] = False
            else:
                mask = tlx.scatter_update(mask, neg_idx, tlx.zeros((neg_idx.shape[0],), dtype=tlx.bool))

    else:
        idx = tlx.to_device(idx, 'cpu')
        for _ in range(3):
            rnd = sample(population, sample_size, True)
            mask = np.isin(rnd, idx)
            if neg_idx is not None:
                mask |= np.isin(rnd, tlx.to_device(neg_idx, 'cpu'))
            mask = tlx.convert_to_tensor(mask, dtype=tlx.bool, device='cpu')
            rnd = tlx.mask_select(rnd, ~mask)
            if neg_idx is None:
                neg_idx = rnd
            else:
                neg_idx = tlx.concat([neg_idx, rnd], axis=0)
            if tlx.convert_to_numpy(neg_idx).size >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break
        neg_idx = tlx.convert_to_tensor(tlx.convert_to_numpy(neg_idx))
    return vector_to_edge_index(neg_idx, num_nodes, bipartite, force_undirected)


def sample(population, k, cpu=False):
    if population <= k:
        if cpu:
            return tlx.convert_to_tensor(np.arange(population), device='cpu')
        else:
            return tlx.convert_to_tensor(np.arange(population))
    else:
        if cpu:
            return tlx.convert_to_tensor(random.sample(range(population), k), device='cpu')
        else:
            return tlx.convert_to_tensor(random.sample(range(population), k))


def edge_index_to_vector(edge_index, size, bipartite, force_undirected):
    row, col = edge_index

    if bipartite:
        return row * size[1] + col, size[0] * size[1]
    
    elif force_undirected:
        assert size[0] == size[1]
        num_nodes = size[0]

        mask = row < col
        row, col = tlx.mask_select(row, mask), tlx.mask_select(col, mask)
        offset = tlx.gather(tlx.cumsum(tlx.arange(1, num_nodes)), row)
        idx = row * num_nodes + col - offset
        population = (num_nodes * (num_nodes + 1)) // 2 - num_nodes
        return idx, population
    
    else:
        assert size[0] == size[1]
        num_nodes = size[0]

        mask = row != col
        row, col = tlx.mask_select(row, mask), tlx.mask_select(col, mask)
        indice = tlx.mask_select(tlx.arange(0, col.shape[0]), row < col)
        col = tlx.scatter_update(col, indice, tlx.gather(col, indice) - 1)
        
        idx = row * (num_nodes - 1) + col
        population = num_nodes * (num_nodes - 1)
        return idx, population


def vector_to_edge_index(idx, size, bipartite, force_undirected):
    if bipartite:
        row = idx // size[1]
        col = idx % size[1]
        return tlx.stack([row, col])
    
    else:
        assert size[0] == size[1]
        num_nodes = size[0]

        if force_undirected:
            offset = tlx.cumsum(tlx.arange(1, num_nodes))
            end = tlx.arange(num_nodes, num_nodes * num_nodes, num_nodes)
            row = tlx.convert_to_tensor(np.digitize(idx, end - offset, right=True))
            col = (tlx.gather(offset, row) + idx) % num_nodes
            return tlx.stack([tlx.concat([row, col]), tlx.concat([col, row])])
        
        else:
            row = idx // (num_nodes - 1)
            col = idx % (num_nodes - 1)
            indice = tlx.mask_select(tlx.arange(0, col.shape[0]), row <= col)
            col = tlx.scatter_update(col, indice, tlx.gather(col, indice) + 1)
            return tlx.stack([row, col])
