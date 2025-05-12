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
BACKEND = tlx.BACKEND


class PreLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(PreLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        
    def forward(self, data, label=None):
        features = data['features']
        adj_norm = data['adj_norm']
        n_nodes = features.shape[0]
        adj_label = data['adj_label']
        pos_weight = data['pos_weight']
        norm = data['norm']
        
        hidden, recovered, mu, logvar, z = self._backbone.encoder(features, adj_norm)
        
        preds = self._backbone.encoder.dc(z)
        re_loss = self._loss_fn(preds=preds, labels=adj_label, n_nodes=n_nodes,
                                norm=norm, pos_weight=pos_weight, mu=mu, logvar=logvar)
        
        ssl_loss = self._backbone.get_ssl_loss_stage_one(hidden)
        loss = data['w_ssl'] * ssl_loss + re_loss
        return loss
            

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
        

        fusion_emb, ssl_embs, nodes_weight = self._backbone.FeatureFusionForward(z_embeddings)

        losspq, p, q = pq_loss_func(fusion_emb, cluster_centers)
        
        selected_idx_tensor = tlx.convert_to_tensor(selected_idx, dtype=tlx.int64)
        q_selected = tlx.gather(q, selected_idx_tensor)
        align_labels_selected = tlx.gather(align_labels, selected_idx_tensor)
        
        sup_loss = tlx.losses.softmax_cross_entropy_with_logits(
            tlx.log(q_selected + 1e-8),
            align_labels_selected
        )
        
        ssl_loss = 0
        for ix, ssl in enumerate(self._backbone.ssl_agent):
            ssl.set_train()
            ssl_loss += ssl.make_loss_stage_two(z_embeddings, adj_norm)
        
        adj_sim_pred = sample_sim(fusion_emb)
        sim_loss = sim_loss_func(adj_sim_pred, adj_label)
        
        loss = sup_loss + sim_loss + self.w_pq * losspq + self.w_ssl * ssl_loss
        
        data['loss_components'] = {
            'losspq': losspq,
            'sup_loss': sup_loss,
            'ssl_loss': ssl_loss,
            'sim_loss': sim_loss
        }
        data['fusion_emb'] = fusion_emb
        data['q'] = q
        
        return loss


def evaluate_clustering(model, embeddings, labels, num_clusters):
    model.set_eval()
    x = tlx.ops.convert_to_tensor(tlx.ops.convert_to_numpy(embeddings))
    kmeans_input = tlx.convert_to_numpy(x)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(kmeans_input)
    pred = kmeans.predict(kmeans_input)  
    acc, nmi, f1 = cluster_accuracy(pred, labels, num_clusters)

    return acc, nmi, f1, pred, kmeans

def sample_sim(fusion_emb):
    fusion_emb = tlx.convert_to_tensor(fusion_emb, dtype=tlx.float32)
    
    def scale(z):
        zmax = tlx.reduce_max(z, axis=1, keepdims=True)
        zmin = tlx.reduce_min(z, axis=1, keepdims=True)
        denominator = tlx.maximum(zmax - zmin, tlx.convert_to_tensor(1e-12, dtype=tlx.float32))
        z_std = (z - zmin) / denominator
        return z_std
        
    fusion_scaled = scale(fusion_emb)
    squared = tlx.ops.square(fusion_scaled)
    sum_squared = tlx.reduce_sum(squared, axis=1, keepdims=True)
    norm = tlx.sqrt(sum_squared)
    norm = tlx.maximum(norm, tlx.convert_to_tensor(1e-12, dtype=tlx.float32))
    fusion_norm = fusion_scaled / norm
    sim_adj = tlx.matmul(fusion_norm, tlx.transpose(fusion_norm))
    return sim_adj


def generate_pseudo_labels_percentile(mu_z, cluster_center_z, data, cluster_num, per):
    mu_z_np = tlx.convert_to_numpy(mu_z)
    cluster_center_z_np = tlx.convert_to_numpy(cluster_center_z)
    mu_z = tlx.convert_to_tensor(mu_z_np)
    cluster_center_z = tlx.convert_to_tensor(cluster_center_z_np)
    _, p, q_z = pq_loss_func(mu_z, cluster_center_z)
    cluster_pred_score, pred_labels_z = dist_2_label(q_z)
    cluster_pred_score_np = tlx.convert_to_numpy(cluster_pred_score)
    tau_p = np.percentile(cluster_pred_score_np, per*100)
    selected_idx_z_mask = cluster_pred_score >= tau_p
    selected_idx_z_mask_np = tlx.convert_to_numpy(selected_idx_z_mask)
    selected_idx_z = np.where(selected_idx_z_mask_np)[0]
    
    print("The number of pseudo labels:", len(selected_idx_z))
    
    pred_labels_z_np = tlx.convert_to_numpy(pred_labels_z)
    pesu_acc, pesu_nmi, pesu_f1 = cluster_accuracy(pred_labels_z_np[selected_idx_z], data.labels[selected_idx_z],
                                                   cluster_num)
    print('Pesudo labels accuracy: {:.2f},{:.2f},{:.2f}'.format(pesu_acc * 100, pesu_nmi * 100, pesu_f1 * 100))
    print("-----------------------------------------------------------------------------")
    return selected_idx_z, pred_labels_z

def get_adj_label(adj):
    sim_matrix = adj.tocsr()
    sim_label = sim_matrix + sp.eye(sim_matrix.shape[0])
    adj_dense = tlx.convert_to_tensor(sim_label.toarray(), dtype=tlx.float32)
    return adj_dense


def set_seed(seed):

    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    tlx.set_seed(SEED)  

def main(args):
    set_seed(args.seed)
    print(args)

    data = get_dataset(args.dataset_path,args.dataset, True)
    feat_dim = data.features.shape[1]
    cluster_num = np.max(data.labels) + 1
    n_nodes = data.features.shape[0]

    encoder = VGAE(feat_dim, args.hid_dim[0], args.hid_dim[1], 0.0, cluster_num)
    modeldis = Discriminator(args.hid_dim[0], args.hid_dim[1], args.hid_dim[2])
    
    set_of_ssl = ['PairwiseAttrSim', 'PairwiseDistance', 'Par', 'Clu', 'DGI']
    # set_of_ssl = ['Par']

    if data.adj.shape[0] > 5000:
        print("use the DGISample")
        local_set_of_ssl = [ssl if ssl != 'DGI' else 'DGISample' for ssl in set_of_ssl]
    else:
        local_set_of_ssl = set_of_ssl

    model = MoeSSL(data, encoder, modeldis, local_set_of_ssl, args)
    loss_fn = re_loss_func 
    loss_func = PreLoss(model, loss_fn)
    
    if args.use_ckpt == 1:
        print("load the ssl train ARVGA model")
        save_path = f'./pretrained/TotalSSL_{args.encoder}_{args.dataset}_{args.hid_dim[0]}.npz'
        tlx.files.load_and_assign_npz(save_path, model.encoder)

    else:
        model.set_train()
        
        pretrain_data = {
            'stage': 'pretrain',
            'features': model.processed_data.features,
            'adj_norm': model.processed_data.adj_norm,
            'adj_label': model.processed_data.adj_label,
            'pos_weight': model.processed_data.pos_weight,
            'norm': model.processed_data.norm,
            'w_ssl': args.w_ssl_stage_one
        }
        
        optimizer_pretrain = tlx.optimizers.Adam(lr=args.lr_pretrain, weight_decay=0.0)
        
        train_weights = []
        for weight in model.params:
            train_weights.append(weight)        
        train_one_step_pretrain = TrainOneStep(loss_func, optimizer_pretrain, train_weights)
        
        for epoch in range(args.pretrain_epochs):
            train_loss = train_one_step_pretrain(pretrain_data, None)
            print(f'Epoch {epoch}, pretrain loss: {train_loss.item()}')
        
        if args.save_ckpt == 1:
            save_path = f"./pretrained/TotalSSL_{args.encoder}_{args.dataset}_{args.hid_dim[0]}.npz"
            tlx.files.save_npz(model.encoder.all_weights, save_path)

    model.encoder.set_eval()
    bottom_embeddings, _, mu_z, _, _ = model.encoder(model.processed_data.features, model.processed_data.adj_norm)
    
    acc, nmi, f1, mu_pred, kmeans = evaluate_clustering(model, mu_z, data.labels, cluster_num)
    print('Z-embddings clustering result: {:.2f}  {:.2f}  {:.2f}'.format(acc * 100, nmi * 100, f1 * 100))
    
    z_embeddings = tlx.ops.convert_to_tensor(tlx.ops.convert_to_numpy(mu_z))

    cluster_center_z = tlx.convert_to_tensor(kmeans.cluster_centers_)
    selected_idx_z, pred_labels_z = generate_pseudo_labels_percentile(z_embeddings, cluster_center_z,
                                                                     data, cluster_num, args.st_per_p)
    
    train_weights = list(model.gate.trainable_weights)
    for ix, agent in enumerate(model.ssl_agent):
        if agent.disc2 is not None:
            train_weights = train_weights + list(agent.disc2.trainable_weights)
        if hasattr(agent, 'gcn2'):
            train_weights = train_weights + list(agent.gcn2.trainable_weights)
    
    adj_label = get_adj_label(data.adj)
    
    best_acc = 0
    best_nmi = 0
    best_f1 = 0
    best_epoch = 0
    best_pred = None


    model.set_train()
    fusion_emb, _, _ = model.FeatureFusionForward(z_embeddings)
    

    acc, nmi, f1, moe_pred_labels, kmeans = evaluate_clustering(model, fusion_emb, data.labels, cluster_num)
    

    pred_labels_z_np = tlx.convert_to_numpy(pred_labels_z)
    pseudo_labels_onehot_np = np.zeros((n_nodes, cluster_num))
    for i in range(n_nodes):
        pseudo_labels_onehot_np[i, pred_labels_z_np[i]] = 1.0
    pseudo_labels_onehot = tlx.convert_to_tensor(pseudo_labels_onehot_np, dtype=tlx.float32)
    pseudo_labels_onehot_cpu = tlx.convert_to_numpy(pseudo_labels_onehot)
    if not isinstance(moe_pred_labels, np.ndarray):
        moe_pred_labels_cpu = tlx.convert_to_numpy(moe_pred_labels)
    else:
        moe_pred_labels_cpu = moe_pred_labels
    align_labels_z_np = cluster_alignment_simple(data, moe_pred_labels_cpu, [pseudo_labels_onehot_cpu])[0]
    align_labels_z = tlx.convert_to_tensor(align_labels_z_np)
    
    cluster_centers_np = kmeans.cluster_centers_
    cluster_centers = tlx.nn.Parameter(
        tlx.convert_to_tensor(cluster_centers_np, dtype=tlx.float32),
        name="cluster_centers"
    )


    train_weights.append(cluster_centers)
    
    loss_func_train = DyFSSTrainLoss(model, args.w_pq, args.w_ssl_stage_two)
    
    if BACKEND == 'tensorflow':
        clean_weights = []
        for weight in train_weights:
            if hasattr(weight, 'name') and '/' in weight.name:
                import tensorflow as tf
                new_name = weight.name.replace('/', '_')
                new_var = tf.Variable(weight.value(), name=new_name)
                clean_weights.append(new_var)
            else:
                clean_weights.append(weight)
        train_weights = clean_weights
    else:
        pass

    optimizer_train = tlx.optimizers.Adam(lr=args.lr_train_fusion, weight_decay=0.0)
    
    train_one_step = TrainOneStep(loss_func_train, optimizer_train, train_weights)
    
    train_data = {
        'z_embeddings': z_embeddings,
        'adj_norm': model.processed_data.adj_norm,
        'cluster_centers': cluster_centers,
        'selected_idx': selected_idx_z,
        'align_labels': align_labels_z,
        'adj_label': adj_label
    }
    
    for epoch in range(args.labels_epochs):
        print(f"Epoch {epoch}")
        model.set_train()
        
        fusion_emb, _, _ = model.FeatureFusionForward(z_embeddings)
        
        if epoch == 0:
            acc, nmi, f1, moe_pred_labels, kmeans = evaluate_clustering(model, fusion_emb, data.labels, cluster_num)
            print('Initial fusion_emb cluster result: {:.2f}  {:.2f}  {:.2f}'.format(acc * 100, nmi * 100, f1 * 100))
            
            if cluster_centers is None:
                cluster_centers = tlx.nn.Parameter(
                    tlx.convert_to_tensor(kmeans.cluster_centers_, dtype=tlx.float32),
                    name="cluster_centers"
                )
                train_weights.append(cluster_centers)
                
                if optimizer_train is None:
                    optimizer_train = tlx.optimizers.Adam(lr=args.lr_train_fusion, weight_decay=0.0)
                    train_one_step = TrainOneStep(loss_func_train, optimizer_train, train_weights)
        
        adj_sim_pred = sample_sim(fusion_emb)
        
        losspq, p, q = pq_loss_func(fusion_emb, cluster_centers)
        
        selected_idx_tensor = tlx.convert_to_tensor(selected_idx_z, dtype=tlx.int64)
        q_selected = tlx.gather(q, selected_idx_tensor)
        align_labels_selected = tlx.gather(align_labels_z, selected_idx_tensor)
        
        sup_loss = tlx.losses.softmax_cross_entropy_with_logits(
            tlx.log(q_selected + 1e-8),
            align_labels_selected
        )
        ssl_loss = model.get_ssl_loss_stage_two(z_embeddings, model.processed_data.adj_norm)
        sim_loss = sim_loss_func(adj_sim_pred, adj_label)
        loss = sup_loss + sim_loss + args.w_pq * losspq + args.w_ssl_stage_two * ssl_loss
        
        loss_np = tlx.convert_to_numpy(loss)
        losspq_np = tlx.convert_to_numpy(losspq)
        sup_loss_np = tlx.convert_to_numpy(sup_loss)
        ssl_loss_np = tlx.convert_to_numpy(ssl_loss)
        sim_loss_np = tlx.convert_to_numpy(sim_loss)
        
        print(
            "loss pq: {:.4f}, pos_loss: {:.4f}, ssl_loss: {:.4f}, sim_loss: {:.4f}".format(
                float(losspq_np), float(sup_loss_np), float(ssl_loss_np), float(sim_loss_np)
            )
        )
        
        train_data = {
            'z_embeddings': z_embeddings,
            'adj_norm': model.processed_data.adj_norm,
            'cluster_centers': cluster_centers,
            'selected_idx': selected_idx_z,
            'align_labels': align_labels_z,
            'adj_label': adj_label
        }
        train_loss = train_one_step(train_data, None)
        model.set_eval()
        fusion_emb, _, _ = model.FeatureFusionForward(z_embeddings)
        
        acc, nmi, f1, cluster_pred, _ = evaluate_clustering(model, fusion_emb, data.labels, cluster_num)
        print('Evaluate clustering result: {:.2f},{:.2f},{:.2f}'.format(acc * 100, nmi * 100, f1 * 100))
        
        if acc > best_acc:
            best_acc = acc
            best_nmi = nmi
            best_f1 = f1
            best_epoch = epoch
            best_pred = cluster_pred
    
    print("Best result at epoch {}: ACC={:.2f}, NMI={:.2f}, F1={:.2f}".format(
        best_epoch, best_acc * 100, best_nmi * 100, best_f1 * 100))
    
    print("Training ending...")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='use cpu or gpu.')
    parser.add_argument('--gpu', type=int, default=1, help='GPU id.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='photo', help='Type of dataset.')
    parser.add_argument("--dataset_path", type=str, default=r'./data', help="path to save dataset")
    parser.add_argument('--encoder', type=str, default="ARVGA", help='the model name')
    parser.add_argument('--hid_dim', type=int, nargs='+', default=[256, 128, 512])
    parser.add_argument('--lr_pretrain', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--pretrain_epochs', type=int, default=1000, help='Number of pretrained totalssl epochs.')
    parser.add_argument('--w_ssl_stage_one', type=float, default=0.25, help='the ssl_loss weight of pretrain stage one')
    parser.add_argument('--use_ckpt', type=int, default=0, help='whether to use checkpoint, 0/1.')
    parser.add_argument('--save_ckpt', type=int, default=1, help='whether to save checkpoint, 0/1.')
    parser.add_argument('--top_k', type=int, default=2, help='The number of experts to choose.')
    parser.add_argument('--st_per_p', type=float, default=0.5, help='The threshold of the pseudo positive labels.')
    parser.add_argument('--lr_train_fusion', type=float, default=0.001, help='train pseudo labels learning rate.')
    parser.add_argument('--labels_epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--w_pq', type=float, default=0.05, help='weight of loss pq.')
    parser.add_argument('--w_ssl_stage_two', type=float, default=0.01, help='the ssl_loss weight of train stage')
    args = parser.parse_args()
    opt = vars(args)

    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")
        
    main(args)

