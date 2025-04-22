# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   dyfss_trainer.py
@Time    :   2025/03/31 12:12:55
@Author  :   zhangliwen
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR

import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
import random
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss
from examples.dyfss.utils.cluster_alignment import *
from examples.dyfss.utils.metric import cluster_accuracy
from examples.dyfss.utils.losses import pq_loss_func, dist_2_label, sim_loss_func,re_loss_func
from examples.dyfss.utils.utils import get_dataset, weights_init, sparse_to_tuple
from gammagl.models.dyfss import VGAE, Discriminator,MoeSSL,InnerProductDecoder

# 添加BACKEND变量定义
BACKEND = tlx.BACKEND
print(f"当前使用的后端: {BACKEND}")


# 定义损失类
class DyFSSLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(DyFSSLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        
    def forward(self, data, label=None):
        # 第一阶段预训练损失
        if isinstance(data, dict) and 'stage' in data and data['stage'] == 'pretrain':
            features = data['features']
            adj_norm = data['adj_norm']
            n_nodes = features.shape[0]
            adj_label = data['adj_label']
            pos_weight = data['pos_weight']
            norm = data['norm']
            
            hidden, recovered, mu, logvar, z = self._backbone.encoder(features, adj_norm)
            
            # 重构损失
            preds = self._backbone.encoder.dc(z)
            re_loss = self._loss_fn(preds=preds, labels=adj_label, n_nodes=n_nodes,
                                   norm=norm, pos_weight=pos_weight, mu=mu, logvar=logvar)
            
            # SSL损失
            ssl_loss = self._backbone.get_ssl_loss_stage_one(hidden)
            loss = data['w_ssl'] * ssl_loss + re_loss
            return loss
            
        # 第二阶段训练损失
        elif isinstance(data, dict) and 'stage' in data and data['stage'] == 'train':
            z_embeddings = data['z_embeddings']
            adj_norm = data['adj_norm']
            cluster_centers = data['cluster_centers']
            selected_idx = data['selected_idx']
            align_labels = data['align_labels']
            adj_label = data['adj_label']
            w_pq = data['w_pq']
            w_ssl = data['w_ssl']
            
            # 特征融合前向传播
            fusion_emb, ssl_embs, nodes_weight = self._backbone.FeatureFusionForward(z_embeddings)
            
            # 计算PQ损失
            losspq, p, q = pq_loss_func(fusion_emb, cluster_centers)
            
            # 计算监督损失
            selected_idx_tensor = tlx.convert_to_tensor(selected_idx, dtype=tlx.int64)
            q_selected = tlx.gather(q, selected_idx_tensor)
            align_labels_selected = tlx.gather(align_labels, selected_idx_tensor)
            
            # 计算交叉熵损失
            sup_loss = tlx.losses.softmax_cross_entropy_with_logits(
                tlx.log(q_selected + 1e-8),
                align_labels_selected
            )
            
            # 计算SSL损失
            ssl_loss = 0
            for ix, ssl in enumerate(self._backbone.ssl_agent):
                ssl.set_train()
                ssl_loss += ssl.make_loss_stage_two(z_embeddings, adj_norm)
            
            # 计算相似度损失
            adj_sim_pred = sample_sim(fusion_emb)
            sim_loss = sim_loss_func(adj_sim_pred, adj_label)
            
            # 总损失
            loss = sup_loss + sim_loss + w_pq * losspq + w_ssl * ssl_loss
            
            # 保存损失组件和中间结果供评估使用
            data['loss_components'] = {
                'losspq': losspq,
                'sup_loss': sup_loss,
                'ssl_loss': ssl_loss,
                'sim_loss': sim_loss
            }
            data['fusion_emb'] = fusion_emb
            data['q'] = q
            
            return loss
        
        else:
            raise ValueError("Unknown stage in data")

# 定义第二阶段的损失类
class DyFSSTrainLoss(WithLoss):
    def __init__(self, net, w_pq, w_ssl):
        super(DyFSSTrainLoss, self).__init__(backbone=net, loss_fn=None)
        self.w_pq = w_pq
        self.w_ssl = w_ssl
        
    def forward(self, data, label=None):
        z_embeddings = data['z_embeddings']
        adj_norm = data['adj_norm']
        cluster_centers = data['cluster_centers']
        selected_idx = data['selected_idx']
        align_labels = data['align_labels']
        adj_label = data['adj_label']
        
        # 特征融合前向传播
        fusion_emb, ssl_embs, nodes_weight = self._backbone.FeatureFusionForward(z_embeddings)
        
        # 计算PQ损失
        losspq, p, q = pq_loss_func(fusion_emb, cluster_centers)
        
        # 计算监督损失
        selected_idx_tensor = tlx.convert_to_tensor(selected_idx, dtype=tlx.int64)
        q_selected = tlx.gather(q, selected_idx_tensor)
        align_labels_selected = tlx.gather(align_labels, selected_idx_tensor)
        
        # 计算交叉熵损失
        sup_loss = tlx.losses.softmax_cross_entropy_with_logits(
            tlx.log(q_selected + 1e-8),
            align_labels_selected
        )
        
        # 计算SSL损失
        ssl_loss = 0
        for ix, ssl in enumerate(self._backbone.ssl_agent):
            ssl.set_train()
            ssl_loss += ssl.make_loss_stage_two(z_embeddings, adj_norm)
        
        # 计算相似度损失
        adj_sim_pred = sample_sim(fusion_emb)
        sim_loss = sim_loss_func(adj_sim_pred, adj_label)
        
        # 总损失
        loss = sup_loss + sim_loss + self.w_pq * losspq + self.w_ssl * ssl_loss
        
        # 保存损失组件和中间结果供评估使用
        data['loss_components'] = {
            'losspq': losspq,
            'sup_loss': sup_loss,
            'ssl_loss': ssl_loss,
            'sim_loss': sim_loss
        }
        data['fusion_emb'] = fusion_emb
        data['q'] = q
        
        return loss

# 评估函数
def evaluate_clustering(model, embeddings, labels, num_clusters):
    """评估聚类结果"""
    model.set_eval()
    # 使用TensorLayerX的正确API来停止梯度
    x = tlx.ops.convert_to_tensor(tlx.ops.convert_to_numpy(embeddings))
    
    # 转换为NumPy数组
    kmeans_input = tlx.convert_to_numpy(x)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(kmeans_input)
    pred = kmeans.predict(kmeans_input)  # 这里pred已经是NumPy数组
    acc, nmi, f1 = cluster_accuracy(pred, labels, num_clusters)
    
    # 确保返回的pred是NumPy数组，不需要再次转换
    return acc, nmi, f1, pred, kmeans

def sample_sim(fusion_emb):
    """计算样本相似度"""
    # 确保输入是TensorLayerX张量
    fusion_emb = tlx.convert_to_tensor(fusion_emb, dtype=tlx.float32)
    
    def scale(z):
        # 确保计算稳定性
        zmax = tlx.reduce_max(z, axis=1, keepdims=True)
        zmin = tlx.reduce_min(z, axis=1, keepdims=True)
        # 避免除以零
        denominator = tlx.maximum(zmax - zmin, tlx.convert_to_tensor(1e-12, dtype=tlx.float32))
        z_std = (z - zmin) / denominator
        return z_std
        
    fusion_scaled = scale(fusion_emb)
    
    # 计算每行的L2范数
    squared = tlx.ops.square(fusion_scaled)
    sum_squared = tlx.reduce_sum(squared, axis=1, keepdims=True)
    norm = tlx.sqrt(sum_squared)
    # 防止除以零
    norm = tlx.maximum(norm, tlx.convert_to_tensor(1e-12, dtype=tlx.float32))
    # 归一化
    fusion_norm = fusion_scaled / norm
    
    # 使用 tlx 的矩阵乘法
    sim_adj = tlx.matmul(fusion_norm, tlx.transpose(fusion_norm))
    return sim_adj

# 在generate_pseudo_labels_percentile函数中
def generate_pseudo_labels_percentile(mu_z, cluster_center_z, data, cluster_num, per):
    """生成伪标签"""
    # 使用TensorLayerX的统一API处理设备问题
    backend = tlx.BACKEND
    
    # 针对不同后端处理设备问题
    if backend == 'torch':
        # PyTorch后端特有的设备处理
        if hasattr(cluster_center_z, 'device') and hasattr(mu_z, 'device'):
            if str(cluster_center_z.device) != str(mu_z.device):
                cluster_center_z = tlx.convert_to_tensor(tlx.convert_to_numpy(cluster_center_z), device=mu_z.device)
    else:
        # 其他后端直接使用convert_to_tensor确保兼容性
        cluster_center_z = tlx.convert_to_tensor(tlx.convert_to_numpy(cluster_center_z))
        mu_z = tlx.convert_to_tensor(tlx.convert_to_numpy(mu_z))
    
    # 不需要stop_gradient，直接使用张量
    _, p, q_z = pq_loss_func(mu_z, cluster_center_z)
    cluster_pred_score, pred_labels_z = dist_2_label(q_z)
    
    # 转换为NumPy进行百分位数计算
    cluster_pred_score_np = tlx.convert_to_numpy(cluster_pred_score)
    tau_p = np.percentile(cluster_pred_score_np, per*100)
    
    # 创建选择掩码
    selected_idx_z_mask = cluster_pred_score >= tau_p
    selected_idx_z_mask_np = tlx.convert_to_numpy(selected_idx_z_mask)
    
    # 获取索引
    selected_idx_z = np.where(selected_idx_z_mask_np)[0]
    
    print("The number of pseudo labels:", len(selected_idx_z))
    
    # 转换为NumPy进行评估
    pred_labels_z_np = tlx.convert_to_numpy(pred_labels_z)
    pesu_acc, pesu_nmi, pesu_f1 = cluster_accuracy(pred_labels_z_np[selected_idx_z], data.labels[selected_idx_z],
                                                   cluster_num)
    print('Pesudo labels accuracy: {:.2f},{:.2f},{:.2f}'.format(pesu_acc * 100, pesu_nmi * 100, pesu_f1 * 100))
    print("-----------------------------------------------------------------------------")
    return selected_idx_z, pred_labels_z

def get_adj_label(adj):
    """获取邻接标签"""
    # 添加自环
    sim_matrix = adj.tocsr()
    sim_label = sim_matrix + sp.eye(sim_matrix.shape[0])
    
    # 直接转换为稠密张量，而不是使用sparse_to_dense
    adj_dense = tlx.convert_to_tensor(sim_label.toarray(), dtype=tlx.float32)
    return adj_dense

def visualize_clustering_result(embeddings, pred_labels, true_labels, dataset_name, acc, nmi, f1):
    """可视化聚类结果"""
    # Use t-SNE for dimensionality reduction to 2D
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create save directory
    save_dir = os.path.join("./examples/dyfss/visualization")
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot predicted clustering results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=pred_labels, cmap='tab20', alpha=0.7, s=10)
    plt.colorbar(scatter, label='Cluster Labels')
    plt.title(f'Dataset: {dataset_name} Clustering Results\nACC: {acc*100:.2f}%, NMI: {nmi*100:.2f}%, F1: {f1*100:.2f}%')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_cluster_pred.png'), dpi=300)
    plt.close()
    
    # Plot ground truth labels
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=true_labels, cmap='tab20', alpha=0.7, s=10)
    plt.colorbar(scatter, label='Ground Truth Labels')
    plt.title(f'Dataset: {dataset_name} Ground Truth Distribution')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_cluster_true.png'), dpi=300)
    plt.close()
    
    print(f"Visualization results saved to {save_dir}")

def set_seed(seed):
    """设置随机种子"""
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    tlx.set_seed(SEED)  

def main():
    set_seed(args.seed)
    print(args)

    # 加载数据集
    data = get_dataset(args.dataset, True)
    feat_dim = data.features.shape[1]
    cluster_num = np.max(data.labels) + 1
    n_nodes = data.features.shape[0]

    # 初始化编码器和判别器
    encoder = VGAE(feat_dim, args.hid_dim[0], args.hid_dim[1], 0.0, cluster_num)
    modeldis = Discriminator(args.hid_dim[0], args.hid_dim[1], args.hid_dim[2])

    # 设置SSL集合
    # set_of_ssl = ['PairwiseAttrSim', 'PairwiseDistance', 'Par', 'Clu', 'DGI']
    set_of_ssl = ['Par']
    if data.adj.shape[0] > 5000:
        print("use the DGISample")
        local_set_of_ssl = [ssl if ssl != 'DGI' else 'DGISample' for ssl in set_of_ssl]
    else:
        local_set_of_ssl = set_of_ssl

    # 初始化MoeSSL模型
    device = args.device
    model = MoeSSL(data, encoder, modeldis, local_set_of_ssl, args,device)
    
    # 定义损失函数和优化器   
    loss_fn = re_loss_func  # 第一阶段使用重构损失
    loss_func = DyFSSLoss(model, loss_fn)
    
    # 第一阶段：预训练
    if args.use_ckpt == 1:
        print("load the ssl train ARVGA model")
        save_path = f'./examples/dyfss/pretrained/TotalSSL_{args.encoder}_{args.dataset}_{args.hid_dim[0]}.npz'
        try:
            tlx.files.load_and_assign_npz(save_path, model.encoder)
        except:
            print("TensorLayerX加载失败，尝试其他方式")
    else:
        # 预训练阶段
        model.set_train()
        
        # 准备预训练数据
        pretrain_data = {
            'stage': 'pretrain',
            'features': model.processed_data.features,
            'adj_norm': model.processed_data.adj_norm,
            'adj_label': model.processed_data.adj_label,
            'pos_weight': model.processed_data.pos_weight,
            'norm': model.processed_data.norm,
            'w_ssl': args.w_ssl_stage_one
        }
        
        # 预训练优化器
        optimizer_pretrain = tlx.optimizers.Adam(lr=args.lr_pretrain, weight_decay=0.0)
        # 预训练训权重
        train_weights = model.params
        # 预训练训练器
        train_one_step_pretrain = TrainOneStep(loss_func, optimizer_pretrain, train_weights)
        
        # 预训练循环
        for epoch in range(args.pretrain_epochs):
            # 训练
            train_loss = train_one_step_pretrain(pretrain_data, None)
            print(f'Epoch {epoch}, pretrain loss: {train_loss.item()}')
        
        # 保存预训练模型
        if args.save_ckpt == 1:
            save_path = f"./examples/dyfss/pretrained/TotalSSL_{args.encoder}_{args.dataset}_{args.hid_dim[0]}.npz"
            tlx.files.save_npz(model.encoder.all_weights, save_path)

    # 获取嵌入
    model.encoder.set_eval()
    bottom_embeddings, _, mu_z, _, _ = model.encoder(model.processed_data.features, model.processed_data.adj_norm)
    
    # 评估嵌入
    acc, nmi, f1, mu_pred, kmeans = evaluate_clustering(model, mu_z, data.labels, cluster_num)
    print('Z-embddings clustering result: {:.2f}  {:.2f}  {:.2f}'.format(acc * 100, nmi * 100, f1 * 100))
    
    z_embeddings = tlx.ops.convert_to_tensor(tlx.ops.convert_to_numpy(mu_z))

    # 获取伪标签
    cluster_center_z = tlx.convert_to_tensor(kmeans.cluster_centers_)
    selected_idx_z, pred_labels_z = generate_pseudo_labels_percentile(z_embeddings, cluster_center_z,
                                                                     data, cluster_num, args.st_per_p)
    
    # 第二阶段：训练
    print("开始第二阶段训练...")
    
    # 收集需要训练的参数
    train_weights = list(model.gate.trainable_weights)
    for ix, agent in enumerate(model.ssl_agent):
        if agent.disc2 is not None:
            train_weights = train_weights + list(agent.disc2.trainable_weights)
        if hasattr(agent, 'gcn2'):
            train_weights = train_weights + list(agent.gcn2.trainable_weights)
    
    # 获取邻接标签
    adj_label = get_adj_label(data.adj)
    
    # 初始化最佳结果变量
    best_acc = 0
    best_nmi = 0
    best_f1 = 0
    best_epoch = 0
    best_pred = None
    
    # 获取融合嵌入进行初始评估
    model.set_train()
    fusion_emb, _, _ = model.FeatureFusionForward(z_embeddings)
    
    # 初始评估
    acc, nmi, f1, moe_pred_labels, kmeans = evaluate_clustering(model, fusion_emb, data.labels, cluster_num)
    print('Initial fusion_emb cluster result: {:.2f}  {:.2f}  {:.2f}'.format(acc * 100, nmi * 100, f1 * 100))
    
    # 准备对齐标签
    print("准备对齐标签...")
    # 转换为numpy数组处理
    pred_labels_z_np = tlx.convert_to_numpy(pred_labels_z)
    # 使用numpy创建one-hot编码
    pseudo_labels_onehot_np = np.zeros((n_nodes, cluster_num))
    for i in range(n_nodes):
        pseudo_labels_onehot_np[i, pred_labels_z_np[i]] = 1.0
    # 转回TensorLayerX张量
    pseudo_labels_onehot = tlx.convert_to_tensor(pseudo_labels_onehot_np, dtype=tlx.float32)
    
    # 删除以下错误代码，因为已经在上面创建了one-hot编码
    # for i in range(n_nodes):
    #     label_idx = tlx.convert_to_numpy(tenor_pseudo_labels[i]).item()
    #     pseudo_labels_onehot = tlx.tensor_scatter_nd_update(
    #         pseudo_labels_onehot, 
    #         [[i, label_idx]], 
    #         [1.0]
    #     )
    
    pseudo_labels_onehot_cpu = tlx.convert_to_numpy(pseudo_labels_onehot)
    
    
    if not isinstance(moe_pred_labels, np.ndarray):
        moe_pred_labels_cpu = tlx.convert_to_numpy(moe_pred_labels)
    else:
        moe_pred_labels_cpu = moe_pred_labels
    
    
    try:
        print("执行聚类对齐...")
        align_labels_z_np = cluster_alignment_simple(data, moe_pred_labels_cpu, [pseudo_labels_onehot_cpu])[0]
        align_labels_z = tlx.convert_to_tensor(align_labels_z_np)
    except Exception as e:
        print(f"聚类对齐时出错: {e}")
        print("使用原始伪标签作为对齐标签...")
        align_labels_z = pseudo_labels_onehot
    
    
    # 确保集群中心是正确的可训练变量
    cluster_centers_np = kmeans.cluster_centers_
    cluster_centers = tlx.nn.Parameter(
        tlx.convert_to_tensor(cluster_centers_np, dtype=tlx.float32),
        name="cluster_centers"
    )
    # 将变量添加到训练权重列表
    train_weights.append(cluster_centers)

    
    
    # 创建第二阶段的损失函数、优化器、训练器
    loss_func_train = DyFSSTrainLoss(model, args.w_pq, args.w_ssl_stage_two)
    optimizer_train = tlx.optimizers.Adam(lr=args.lr_train_fusion, weight_decay=0.0)
    train_one_step = TrainOneStep(loss_func_train, optimizer_train, train_weights)
    
    # 准备第二阶段训练数据
    train_data = {
        'z_embeddings': z_embeddings,
        'adj_norm': model.processed_data.adj_norm,
        'cluster_centers': cluster_centers,
        'selected_idx': selected_idx_z,
        'align_labels': align_labels_z,
        'adj_label': adj_label
    }
    
    # 第二阶段训练循环
    print("开始第二阶段训练循环...")
    for epoch in range(args.labels_epochs):
        print(f"Epoch {epoch}")
        model.set_train()
        
        try:
            # 训练步骤
            train_loss = train_one_step(train_data, None)
            
            # 获取损失组件
            if 'loss_components' in train_data:
                loss_components = train_data['loss_components']
                # 针对不同后端处理item()方法
                if BACKEND == 'torch':
                    print(
                        "loss pq: {:.4f}, pos_loss: {:.4f}, ssl_loss: {:.4f}, sim_loss: {:.4f}".format(
                            loss_components['losspq'].item(), 
                            loss_components['sup_loss'].item(), 
                            loss_components['ssl_loss'].item(), 
                            loss_components['sim_loss'].item()
                        )
                    )
                else:
                    # 对于非PyTorch后端，直接使用张量值
                    print(
                        "loss pq: {:.4f}, pos_loss: {:.4f}, ssl_loss: {:.4f}, sim_loss: {:.4f}".format(
                            float(tlx.convert_to_numpy(loss_components['losspq'])), 
                            float(tlx.convert_to_numpy(loss_components['sup_loss'])), 
                            float(tlx.convert_to_numpy(loss_components['ssl_loss'])), 
                            float(tlx.convert_to_numpy(loss_components['sim_loss']))
                        )
                    )
            else:
                # 同样处理不同后端的item()方法
                if BACKEND == 'torch':
                    print(f"总损失: {train_loss.item()}")
                else:
                    print(f"总损失: {float(tlx.convert_to_numpy(train_loss))}")
            
            # 评估
            model.set_eval()
            if 'fusion_emb' in train_data:
                fusion_emb = train_data['fusion_emb']
            else:
                fusion_emb, _, _ = model.FeatureFusionForward(z_embeddings)
            
            # 使用kmeans评估
            acc, nmi, f1, cluster_pred, _ = evaluate_clustering(model, fusion_emb, data.labels, cluster_num)
            print('Evaluate clustering result: {:.2f},{:.2f},{:.2f}'.format(acc * 100, nmi * 100, f1 * 100))
            
            # 保存最佳结果
            if acc > best_acc:
                best_acc = acc
                best_nmi = nmi
                best_f1 = f1
                best_epoch = epoch
                best_pred = cluster_pred
                
        except Exception as e:
            print(f"训练过程中出错: {e}")
            # 针对不同后端打印更详细的错误信息
            if BACKEND == 'tensorflow':
                import tensorflow as tf
                if isinstance(e, tf.errors.ResourceExhaustedError):
                    print("TensorFlow内存不足错误，尝试减小批量大小或模型规模")
                elif isinstance(e, tf.errors.InvalidArgumentError):
                    print("TensorFlow参数错误，可能是张量形状不匹配")
            import traceback
            traceback.print_exc()
            print("跳过此轮训练...")
            continue
    
    # 如果没有成功的训练轮次，设置默认值
    if best_pred is None:
        print("警告：没有成功的训练轮次，使用初始评估结果")
        best_epoch = -1
        best_acc = acc
        best_nmi = nmi
        best_f1 = f1
            
    print("Best result at epoch {}: ACC={:.2f}, NMI={:.2f}, F1={:.2f}".format(
        best_epoch, best_acc * 100, best_nmi * 100, best_f1 * 100))
    
    print("Training ending...")
    
    # # 最终评估
    # try:
    #     model.set_eval()
    #     fusion_emb, _, nodes_weight = model.FeatureFusionForward(z_embeddings)
    #     acc, nmi, f1, cluster_pred, _ = evaluate_clustering(model, fusion_emb, data.labels, cluster_num)
    #     print('Final clustering result: {:.2f},{:.2f},{:.2f}'.format(acc * 100, nmi * 100, f1 * 100))
    # except Exception as e:
    #     print(f"最终评估时出错: {e}")
    #     print("使用最佳结果作为最终结果...")
    #     print('Final clustering result: {:.2f},{:.2f},{:.2f}'.format(best_acc * 100, best_nmi * 100, best_f1 * 100))

    # # 可视化结果（可选）
    # visualize_clustering_result(tlx.convert_to_numpy(fusion_emb), 
    #                         tlx.convert_to_numpy(cluster_pred), 
    #                         data.labels,  
    #                         args.dataset,
    #                         acc, nmi, f1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='use cpu or gpu.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='photo', help='Type of dataset.')
    parser.add_argument('--encoder', type=str, default="ARVGA", help='the model name')
    parser.add_argument('--hid_dim', type=int, nargs='+', default=[256, 128, 512])
    parser.add_argument('--lr_pretrain', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--pretrain_epochs', type=int, default=3100, help='Number of pretrained totalssl epochs.')
    parser.add_argument('--w_ssl_stage_one', type=float, default=0.25, help='the ssl_loss weight of pretrain stage one')
    parser.add_argument('--use_ckpt', type=int, default=0, help='whether to use checkpoint, 0/1.')
    parser.add_argument('--save_ckpt', type=int, default=1, help='whether to save checkpoint, 0/1.')
    parser.add_argument('--top_k', type=int, default=1, help='The number of experts to choose.')
    parser.add_argument('--st_per_p', type=float, default=0.5, help='The threshold of the pseudo positive labels.')
    parser.add_argument('--lr_train_fusion', type=float, default=0.001, help='train pseudo labels learning rate.')
    parser.add_argument('--labels_epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--w_pq', type=float, default=0.05, help='weight of loss pq.')
    parser.add_argument('--w_ssl_stage_two', type=float, default=0.01, help='the ssl_loss weight of train stage')
    args = parser.parse_args()
    opt = vars(args)

    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    main()

