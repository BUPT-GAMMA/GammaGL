import random

import numpy as np
import tensorlayerx as tlx

from gammagl.utils import coalesce, degree, remove_self_loops
from .num_nodes import maybe_num_nodes

def negative_sampling(edge_index, num_nodes = None, num_neg_samples = None, method = 'sparse', force_undirected = False):
    r"""Samples random negative edges of a graph given by :attr:`edge_index`.

    parameters
    ----------
    edge_index: tensor
        The edge indices.
    num_nodes: int, tuple[int, int], optional
        The number of nodes,
        *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
        If given as a tuple, then :obj:`edge_index` is interpreted as a
        bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
        (default: :obj:`None`)
    num_neg_samples: int, optional
        The (approximate) number of negative
        samples to return.
        If set to :obj:`None`, will try to return a negative edge for every
        positive edge. (default: :obj:`None`)
    method: str, optional
        The method to use for negative sampling,
        *i.e.*, :obj:`"sparse"` or :obj:`"dense"`.
        This is a memory/runtime trade-off.
        :obj:`"sparse"` will work on any graph of any size, while
        :obj:`"dense"` can perform faster true-negative checks.
        (default: :obj:`"sparse"`)
    force_undirected: bool, optional
        If set to :obj:`True`, sampled
        negative edges will be undirected. (default: :obj:`False`)

    """
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
        if tlx.BACKEND == 'paddle':
            mask = tlx.ones((population,), dtype=tlx.int64)
            mask[idx] = 0
        else:
            mask = tlx.ones((population,), dtype=tlx.bool)
            mask = tlx.scatter_update(mask, idx, tlx.zeros((idx.shape[0],), dtype=tlx.bool))
        for _ in range(3):
            rnd = sample(population, sample_size)
            if tlx.BACKEND == 'paddle':
                indices = tlx.convert_to_tensor(tlx.convert_to_numpy(tlx.gather(mask, rnd)), dtype = tlx.bool)
            else:
                indices = tlx.gather(mask, rnd)
            rnd = tlx.mask_select(rnd, indices)
            if neg_idx is None:
                neg_idx = rnd
            else:
                neg_idx = tlx.concat([neg_idx, rnd], axis=0)
            if tlx.convert_to_numpy(neg_idx).size >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break
            if tlx.BACKEND == 'paddle':
                mask[neg_idx] = 0
            else:
                mask = tlx.scatter_update(mask, neg_idx, tlx.zeros((neg_idx.shape[0],), dtype=tlx.bool))

    else:
        idx = tlx.to_device(idx, 'cpu')
        for _ in range(3):
            rnd = sample(population, sample_size, True)
            mask = np.isin(rnd, idx)
            if neg_idx is not None:
                mask |= np.isin(tlx.convert_to_numpy(rnd), tlx.convert_to_numpy(neg_idx))
            mask = tlx.convert_to_tensor(mask, dtype=tlx.bool, device='cpu')
            rnd = tlx.mask_select(rnd, ~mask)
            if neg_idx is None:
                neg_idx = rnd
            else:
                neg_idx = tlx.concat([neg_idx, rnd], axis=0)
            if tlx.convert_to_numpy(neg_idx).size >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break
        neg_idx = tlx.convert_to_tensor(tlx.convert_to_numpy(neg_idx), dtype = tlx.int64)
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
        row = tlx.convert_to_numpy(tlx.mask_select(row, mask))
        col = tlx.convert_to_numpy(tlx.mask_select(col, mask))

        indice = tlx.mask_select(tlx.arange(0, col.shape[0]), tlx.convert_to_tensor(row < col))
        indice = tlx.convert_to_numpy(indice)
        col[indice] = col[indice] - 1
        # col = tlx.tensor_scatter_nd_update(col, indice, tlx.gather(col, indice) - 1)
        
        idx = row * (num_nodes - 1) + col
        population = num_nodes * (num_nodes - 1)
        return tlx.convert_to_tensor(idx), population


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
            row = tlx.convert_to_numpy(row)
            col = tlx.convert_to_numpy(col)
            indice = tlx.mask_select(tlx.arange(0, col.shape[0]), tlx.convert_to_tensor(row <= col))
            indice = tlx.convert_to_numpy(indice)
            col[indice] = col[indice] + 1
            row = tlx.convert_to_tensor(row)
            col = tlx.convert_to_tensor(col)
            # col = tlx.scatter_update(col, indice, tlx.gather(col, indice) + 1)
            return tlx.stack([row, col])
