
from sklearn.cluster import KMeans
import numpy as np
import tensorlayerx as tlx


# 修改 re_loss_func 函数，确保在所有后端都能正常工作
def re_loss_func(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    # TensorLayerX没有binary_cross_entropy_with_logits，使用binary_cross_entropy替代
    # 先应用sigmoid激活函数，然后使用binary_cross_entropy
    preds_sigmoid = tlx.sigmoid(preds)
    
    # 确保pos_weight是标量，避免广播操作可能导致的内存问题
    if hasattr(pos_weight, 'shape') and len(pos_weight.shape) > 0:
        pos_weight = pos_weight[0]
    
    # 手动实现带权重的二元交叉熵，使用更稳定的计算方式
    # 分别计算正样本和负样本的损失，避免大型矩阵的直接乘法
    pos_loss = -tlx.log(preds_sigmoid + 1e-8) * pos_weight
    neg_loss = -tlx.log(1 - preds_sigmoid + 1e-8)
    
    # 根据标签选择相应的损失
    loss = labels * pos_loss + (1 - labels) * neg_loss
    
    # 计算平均损失
    cost = norm * tlx.reduce_mean(loss)

    # Check if the model is simple Graph Auto-encoder
    if logvar is None:
        return cost

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # 使用 TensorLayerX 的操作替代 PyTorch 操作
    KLD = -0.5 / n_nodes * tlx.reduce_mean(tlx.reduce_sum(
        1 + 2 * logvar - tlx.pow(mu, 2) - tlx.pow(tlx.exp(logvar), 2), axis=1))
    return cost + KLD


def dis_loss(real, d_fake):
    # 使用 TensorLayerX 的操作替代 PyTorch 操作
    # 为真实样本创建全1标签
    ones_like_real = tlx.ones_like(real)
    # 为假样本创建全0标签
    zeros_like_fake = tlx.zeros_like(d_fake)
    
    # 计算真实样本的二元交叉熵损失
    real_sigmoid = tlx.sigmoid(real)
    dc_loss_real = -tlx.reduce_mean(ones_like_real * tlx.log(real_sigmoid + 1e-8) + 
                                   (1 - ones_like_real) * tlx.log(1 - real_sigmoid + 1e-8))
    
    # 计算假样本的二元交叉熵损失
    fake_sigmoid = tlx.sigmoid(d_fake)
    dc_loss_fake = -tlx.reduce_mean(zeros_like_fake * tlx.log(fake_sigmoid + 1e-8) + 
                                   (1 - zeros_like_fake) * tlx.log(1 - fake_sigmoid + 1e-8))
    
    return dc_loss_real + dc_loss_fake


def pq_loss_func(feat, cluster_centers):
    alpha = 1.0
    # 使用 TensorLayerX 的操作替代 PyTorch 操作
    # 将 feat 扩展一个维度，以便与 cluster_centers 进行广播运算
    feat_expanded = tlx.expand_dims(feat, axis=1)
    # 计算特征与聚类中心的欧氏距离平方
    squared_diff = tlx.pow(feat_expanded - cluster_centers, 2)
    # 在最后一个维度上求和
    squared_distance = tlx.reduce_sum(squared_diff, axis=2)
    # 计算 q 值
    q = 1.0 / (1.0 + squared_distance / alpha)
    q = tlx.pow(q, (alpha + 1.0) / 2.0)
    
    # 计算归一化的 q 值
    q_sum = tlx.reduce_sum(q, axis=1, keepdims=True)
    q = q / q_sum
    
    # 计算权重
    weight = tlx.pow(q, 2)
    weight_sum = tlx.reduce_sum(weight, axis=0, keepdims=True)
    weight = weight / weight_sum
    
    # 计算 p 值
    p_sum = tlx.reduce_sum(weight, axis=1, keepdims=True)
    p = weight / p_sum
    
    # 切断梯度传播
    p = tlx.convert_to_tensor(tlx.convert_to_numpy(p))
    
    # 计算 KL 散度
    log_q = tlx.log(q + 1e-8)
    # 在 TensorLayerX 中实现 KL 散度
    kl_div = tlx.reduce_sum(p * (tlx.log(p + 1e-8) - log_q))
    
    return kl_div, p, q


def dist_2_label(q_t):
    # 使用TensorLayerX的操作替代PyTorch的操作
    # torch.max替换为tlx.argmax和tlx.reduce_max
    maxlabel = tlx.reduce_max(q_t, axis=1)
    label = tlx.argmax(q_t, axis=1)
    return maxlabel, label


def sim_loss_func(adj_preds, adj_labels, weight_tensor=None):
    cost = 0.
    # 手动实现带权重的二元交叉熵
    if weight_tensor is None:
        # 先应用sigmoid激活函数
        preds_sigmoid = tlx.sigmoid(adj_preds)
        
        # 计算二元交叉熵
        bce = -adj_labels * tlx.log(preds_sigmoid + 1e-8) - (1 - adj_labels) * tlx.log(1 - preds_sigmoid + 1e-8)
        
        # 计算平均损失
        cost += tlx.reduce_mean(bce)
    else:
        # 处理稀疏张量转换为密集张量
        adj_labels_dense = adj_labels
        if hasattr(adj_labels, 'to_dense'):
            adj_labels_dense = adj_labels.to_dense()
        elif hasattr(adj_labels, 'todense'):  # 处理scipy稀疏矩阵
            adj_labels_dense = tlx.convert_to_tensor(adj_labels.todense())
        
        # 展平张量
        adj_preds_flat = tlx.reshape(adj_preds, [-1])
        adj_labels_flat = tlx.reshape(adj_labels_dense, [-1])
        
        # 应用sigmoid激活函数
        preds_sigmoid = tlx.sigmoid(adj_preds_flat)
        
        # 计算带权重的二元交叉熵
        bce = -adj_labels_flat * tlx.log(preds_sigmoid + 1e-8) - (1 - adj_labels_flat) * tlx.log(1 - preds_sigmoid + 1e-8)
        
        # 应用权重
        weighted_bce = bce * weight_tensor
        
        # 计算平均损失
        cost += tlx.reduce_mean(weighted_bce)

    return cost