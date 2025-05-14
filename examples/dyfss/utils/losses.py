from sklearn.cluster import KMeans
import numpy as np
import tensorlayerx as tlx



def re_loss_func(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    preds_sigmoid = tlx.sigmoid(preds)
    
    if hasattr(pos_weight, 'shape') and len(pos_weight.shape) > 0:
        pos_weight = pos_weight[0]
    
    pos_loss = -tlx.log(preds_sigmoid + 1e-8) * pos_weight
    neg_loss = -tlx.log(1 - preds_sigmoid + 1e-8)
    
    loss = labels * pos_loss + (1 - labels) * neg_loss
    
    cost = norm * tlx.reduce_mean(loss)

    # Check if the model is simple Graph Auto-encoder
    if logvar is None:
        return cost

    KLD = -0.5 / n_nodes * tlx.reduce_mean(tlx.reduce_sum(
        1 + 2 * logvar - tlx.pow(mu, 2) - tlx.pow(tlx.exp(logvar), 2), axis=1))
    return cost + KLD


def dis_loss(real, d_fake):
    ones_like_real = tlx.ones_like(real)
    zeros_like_fake = tlx.zeros_like(d_fake)
    
    real_sigmoid = tlx.sigmoid(real)
    dc_loss_real = -tlx.reduce_mean(ones_like_real * tlx.log(real_sigmoid + 1e-8) + 
                                   (1 - ones_like_real) * tlx.log(1 - real_sigmoid + 1e-8))
    
    fake_sigmoid = tlx.sigmoid(d_fake)
    dc_loss_fake = -tlx.reduce_mean(zeros_like_fake * tlx.log(fake_sigmoid + 1e-8) + 
                                   (1 - zeros_like_fake) * tlx.log(1 - fake_sigmoid + 1e-8))
    
    return dc_loss_real + dc_loss_fake


def pq_loss_func(feat, cluster_centers):
    alpha = 1.0
    feat_expanded = tlx.expand_dims(feat, axis=1)
    squared_diff = tlx.pow(feat_expanded - cluster_centers, 2)
    squared_distance = tlx.reduce_sum(squared_diff, axis=2)
    q = 1.0 / (1.0 + squared_distance / alpha)
    q = tlx.pow(q, (alpha + 1.0) / 2.0)
    q_sum = tlx.reduce_sum(q, axis=1, keepdims=True)
    q = q / q_sum

    weight = tlx.pow(q, 2)
    weight_sum = tlx.reduce_sum(weight, axis=0, keepdims=True)
    weight = weight / weight_sum
    
    p_sum = tlx.reduce_sum(weight, axis=1, keepdims=True)
    p = weight / p_sum
    p = tlx.convert_to_tensor(tlx.convert_to_numpy(p))

    log_q = tlx.log(q + 1e-8)
    kl_div = tlx.reduce_sum(p * (tlx.log(p + 1e-8) - log_q))
    
    return kl_div, p, q


def dist_2_label(q_t):
    maxlabel = tlx.reduce_max(q_t, axis=1)
    label = tlx.argmax(q_t, axis=1)
    return maxlabel, label


def sim_loss_func(adj_preds, adj_labels, weight_tensor=None):
    cost = 0.
    if weight_tensor is None:
        preds_sigmoid = tlx.sigmoid(adj_preds)
        
        bce = -adj_labels * tlx.log(preds_sigmoid + 1e-8) - (1 - adj_labels) * tlx.log(1 - preds_sigmoid + 1e-8)
        
        cost += tlx.reduce_mean(bce)
    else:
        adj_labels_dense = adj_labels
        if hasattr(adj_labels, 'to_dense'):
            adj_labels_dense = adj_labels.to_dense()
        elif hasattr(adj_labels, 'todense'): 
            adj_labels_dense = tlx.convert_to_tensor(adj_labels.todense())
        
        adj_preds_flat = tlx.reshape(adj_preds, [-1])
        adj_labels_flat = tlx.reshape(adj_labels_dense, [-1])
        
        preds_sigmoid = tlx.sigmoid(adj_preds_flat)
        
        bce = -adj_labels_flat * tlx.log(preds_sigmoid + 1e-8) - (1 - adj_labels_flat) * tlx.log(1 - preds_sigmoid + 1e-8)
        
        weighted_bce = bce * weight_tensor
        
        cost += tlx.reduce_mean(weighted_bce)

    return cost