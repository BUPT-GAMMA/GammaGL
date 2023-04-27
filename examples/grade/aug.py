import tensorlayerx as tlx
from gammagl.utils.corrupt_graph import drop_feature, add_self_loops, mask_edge
from gammagl.data import Graph
import numpy as np


def get_sim(z1, z2):
    '''
    Compute the similarity matix
    '''
    z1 = tlx.ops.l2_normalize(z1, axis=1)
    z2 = tlx.ops.l2_normalize(z2, axis=1)
    return tlx.ops.matmul(z1, tlx.ops.transpose(z2))


def get_node_dist(edge_index, num_nodes):
    """
    Compute adjacent node distribution.
    """
    row, col = edge_index
    dist_list = []
    for i in range(num_nodes):
        dist = tlx.zeros([num_nodes])
        idx = row[(col==i)]
        dist = tlx.scatter_update(dist, idx, tlx.ones_like(idx, dtype=tlx.float32))
        dist_list.append(dist)
    dist_list = tlx.stack(dist_list, axis=0)
    dist_list = tlx.convert_to_tensor(dist_list, dtype=tlx.float32)
    return dist_list


def neighbor_sampling(src_idx, dst_idx, node_dist, sim, max_degree, aug_degree):
    '''
    Change the degree of tail nodes
    '''
    #similar to phi=sim[src_idx,dst_idx] which generates error using tensorflow backend
    src, dst = tlx.convert_to_numpy(src_idx), tlx.convert_to_numpy(dst_idx)
    s = tlx.convert_to_numpy(sim)
    phi = tlx.convert_to_tensor(s[src, dst], dtype=tlx.float32)
    phi = tlx.expand_dims(phi, axis=1)
    #similar to phi = th.clamp(phi, 0, 0.5)
    phi = tlx.minimum(tlx.ones_like(phi)*0.5,tlx.maximum(phi, tlx.zeros_like(phi)))
    mix_dist = tlx.gather(node_dist, dst_idx, axis=0)*phi+tlx.gather(node_dist, src_idx, axis=0)*(1 - phi)
    new_tgt = multinomial(mix_dist+1e-12, int(max_degree))
    tgt_idx =tlx.expand_dims(tlx.arange(0, max_degree), axis=0)
    new_col = new_tgt[(tgt_idx - tlx.expand_dims(aug_degree, axis=1) < 0)]
    new_row = repeat_interleave(src_idx, aug_degree)
    #new_row=src_idx.repeat_interleave(aug_degree)
    return new_row, new_col

def degree_mask_edge(idx, sim, max_degree, node_degree, mask_prob):
    '''
    Change the degree of head nodes
    '''
    #aug_degree = (node_degree * (1- mask_prob)).long()
    aug_degree = tlx.cast(tlx.cast(node_degree, dtype=tlx.float32) * (1 - mask_prob), dtype=tlx.int64)
    aug_degree = tlx.cast(aug_degree, tlx.int64)
    sim_dist = tlx.gather(sim, idx, axis=0)
    new_tgt = multinomial(sim_dist + 1e-12, int(max_degree))
    tgt_idx = tlx.expand_dims(tlx.arange(0, max_degree), axis=0)
    new_col = new_tgt[(tgt_idx - tlx.expand_dims(aug_degree, axis=1) < 0)]
    #new_row=idx.repeat_interleave(aug_degree)
    new_row = repeat_interleave(idx, aug_degree)
    return new_row, new_col

def multinomial(input, num_samples):
    '''
    implement of torch.multinomial()
    '''
    input = tlx.convert_to_numpy(input)
    dim1, dim2 = input.shape[0], input.shape[1]
    input = input /np.sum(input, axis=1, keepdims=True)
    store_sample = []
    for i in range(dim1):
        sample = np.random.choice(dim2, num_samples, p=input[i], replace=False)
        store_sample.append(sample)
    sample = np.stack(store_sample, axis=0)
    sample = tlx.convert_to_tensor(sample, dtype=tlx.int64)
    return sample


def degree_aug(edge_index, x,
               embeds, degree,
               feat_drop_rate_1,
               edge_mask_rate_1,
               feat_drop_rate_2,
               edge_mask_rate_2,
               threshold, num_nodes):
    feat1 = tlx.convert_to_tensor(drop_feature(tlx.convert_to_numpy(x), feat_drop_rate_1))
    feat2 = tlx.convert_to_tensor(drop_feature(tlx.convert_to_numpy(x), feat_drop_rate_2))

    max_degree = np.max(degree)
    node_dist = get_node_dist(edge_index, num_nodes)
    src_idx = tlx.convert_to_tensor(np.argwhere(degree < threshold).flatten(), dtype=tlx.int64)
    rest_idx = tlx.convert_to_tensor(np.argwhere(degree >= threshold).flatten(), dtype=tlx.int64)
    rest_node_degree = degree[degree>=threshold]
    sim = get_sim(embeds, embeds)
    #similar to sim =torch.clamp(sim, 0, 1)
    sim = tlx.minimum(tlx.ones_like(sim), tlx.maximum(sim, tlx.zeros_like(sim)))
    sim = sim-tlx.eye(sim.shape[0])*tlx.diag(sim)
    src_sim = tlx.gather(sim, src_idx, axis=0)

    dst_idx = tlx.squeeze(multinomial(src_sim+1e-12, 1), axis=1)

    #similar to torch_scatter.scatter_add
    rest_node_degree = tlx.convert_to_tensor(rest_node_degree, dtype=tlx.int64)
    degree_dist = tlx.unsorted_segment_sum(tlx.ones(tlx.get_tensor_shape(rest_node_degree)),
                                            rest_node_degree,
                                            num_segments=max_degree+1)
    prob = tlx.expand_dims(degree_dist, axis=0)
    prob = tlx.tile(prob, [tlx.get_tensor_shape(src_idx)[0],1])
    aug_degree = tlx.squeeze(multinomial(prob, 1),axis=1)
    new_row_mix_1, new_col_mix_1 = neighbor_sampling(src_idx, dst_idx, node_dist, sim, max_degree, aug_degree)
    new_row_rest_1, new_col_rest_1 = degree_mask_edge(rest_idx, sim, max_degree, rest_node_degree, edge_mask_rate_1)
    nsrc1 = tlx.concat([new_row_mix_1, new_row_rest_1], axis=0)
    ndst1 = tlx.concat([new_col_mix_1, new_col_rest_1], axis=0)

    edge_index = tlx.stack([nsrc1, ndst1], axis=0)
    edge_index, _ = add_self_loops(edge_index)
    ng1 = Graph(x=feat1, edge_index=edge_index, num_nodes=num_nodes)

    new_row_mix_2, new_col_mix_2 = neighbor_sampling(src_idx, dst_idx, node_dist, sim, max_degree, aug_degree)
    new_row_rest_2, new_col_rest_2 = degree_mask_edge(rest_idx, sim, max_degree, rest_node_degree, edge_mask_rate_2)
    nsrc2 = tlx.concat([new_row_mix_2, new_row_rest_2], axis=0)
    ndst2 = tlx.concat([new_col_mix_2, new_col_rest_2], axis=0)
    edge_index = tlx.stack([nsrc2,ndst2], axis=0)
    edge_index, _ = add_self_loops(edge_index)
    ng2 = Graph(x=feat2,edge_index=edge_index, num_nodes=num_nodes)
    return ng1, ng2



def repeat_interleave(input,repeats):
    '''
    implement of torch.repeat_interleave()
    '''
    input, repeats = tlx.convert_to_numpy(input), tlx.convert_to_numpy(repeats)
    output = tlx.convert_to_tensor(np.repeat(input,repeats), dtype=tlx.int64)
    return output