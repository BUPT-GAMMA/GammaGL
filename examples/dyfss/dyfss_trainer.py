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




class DyFSSLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(DyFSSLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label=None):
        if isinstance(data, dict) and 'stage' in data and data['stage'] == 'pretrain':
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
        
        elif isinstance(data, dict) and 'stage' in data and data['stage'] == 'train':
            z_embeddings = data['z_embeddings']
            adj_norm = data['adj_norm']
            cluster_centers = data['cluster_centers']
            selected_idx = data['selected_idx']
            align_labels = data['align_labels']
            adj_label = data['adj_label']
            w_pq = data['w_pq']
            w_ssl = data['w_ssl']
            
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
            
            loss = sup_loss + sim_loss + w_pq * losspq + w_ssl * ssl_loss
            
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
    """Evaluate clustering results"""
    model.set_eval()
    # Use TensorLayerX's correct API to stop gradients
    x = tlx.ops.convert_to_tensor(tlx.ops.convert_to_numpy(embeddings))

    # Convert to NumPy array
    kmeans_input = tlx.convert_to_numpy(x)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(kmeans_input)
    pred = kmeans.predict(kmeans_input)  # pred is already a NumPy array here
    acc, nmi, f1 = cluster_accuracy(pred, labels, num_clusters)

    # Ensure the returned pred is a NumPy array, no need to convert again
    return acc, nmi, f1, pred, kmeans

def sample_sim(fusion_emb):
    """Calculate sample similarity"""
    # Ensure input is a TensorLayerX tensor
    fusion_emb = tlx.convert_to_tensor(fusion_emb, dtype=tlx.float32)

    def scale(z):
        # Ensure calculation stability
        zmax = tlx.reduce_max(z, axis=1, keepdims=True)
        zmin = tlx.reduce_min(z, axis=1, keepdims=True)
        # Avoid division by zero
        denominator = tlx.maximum(zmax - zmin, tlx.convert_to_tensor(1e-12, dtype=tlx.float32))
        z_std = (z - zmin) / denominator
        return z_std

    fusion_scaled = scale(fusion_emb)

    # Calculate L2 norm for each row
    squared = tlx.ops.square(fusion_scaled)
    sum_squared = tlx.reduce_sum(squared, axis=1, keepdims=True)
    norm = tlx.sqrt(sum_squared)
    # Prevent division by zero
    norm = tlx.maximum(norm, tlx.convert_to_tensor(1e-12, dtype=tlx.float32))
    # Normalize
    fusion_norm = fusion_scaled / norm

    # Use tlx matrix multiplication
    sim_adj = tlx.matmul(fusion_norm, tlx.transpose(fusion_norm))
    return sim_adj

# In generate_pseudo_labels_percentile function
def generate_pseudo_labels_percentile(mu_z, cluster_center_z, data, cluster_num, per):
    """Generate pseudo labels"""
    # Use TensorLayerX's unified API to handle device issues
    backend = tlx.BACKEND

    # Handle device issues for different backends
    if backend == 'torch':
        # PyTorch backend specific device handling
        if hasattr(cluster_center_z, 'device') and hasattr(mu_z, 'device'):
            if str(cluster_center_z.device) != str(mu_z.device):
                cluster_center_z = tlx.convert_to_tensor(tlx.convert_to_numpy(cluster_center_z), device=mu_z.device)
    else:
        # Other backends directly use convert_to_tensor to ensure compatibility
        cluster_center_z = tlx.convert_to_tensor(tlx.convert_to_numpy(cluster_center_z))
        mu_z = tlx.convert_to_tensor(tlx.convert_to_numpy(mu_z))

    # No need for stop_gradient, use tensors directly
    _, p, q_z = pq_loss_func(mu_z, cluster_center_z)
    cluster_pred_score, pred_labels_z = dist_2_label(q_z)

    # Convert to NumPy for percentile calculation
    cluster_pred_score_np = tlx.convert_to_numpy(cluster_pred_score)
    tau_p = np.percentile(cluster_pred_score_np, per*100)

    # Create selection mask
    selected_idx_z_mask = cluster_pred_score >= tau_p
    selected_idx_z_mask_np = tlx.convert_to_numpy(selected_idx_z_mask)

    # Get indices
    selected_idx_z = np.where(selected_idx_z_mask_np)[0]

    print("The number of pseudo labels:", len(selected_idx_z))

    # Convert to NumPy for evaluation
    pred_labels_z_np = tlx.convert_to_numpy(pred_labels_z)
    pesu_acc, pesu_nmi, pesu_f1 = cluster_accuracy(pred_labels_z_np[selected_idx_z], data.labels[selected_idx_z],
                                                   cluster_num)
    print('Pseudo labels accuracy: {:.2f},{:.2f},{:.2f}'.format(pesu_acc * 100, pesu_nmi * 100, pesu_f1 * 100))
    print("-----------------------------------------------------------------------------")
    return selected_idx_z, pred_labels_z

def get_adj_label(adj):
    """Get adjacency labels"""
    # Add self-loops
    sim_matrix = adj.tocsr()
    sim_label = sim_matrix + sp.eye(sim_matrix.shape[0])

    # Directly convert to dense tensor instead of using sparse_to_dense
    adj_dense = tlx.convert_to_tensor(sim_label.toarray(), dtype=tlx.float32)
    return adj_dense


def set_seed(seed):
    """Set random seed"""
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    tlx.set_seed(SEED)

def main():
    set_seed(args.seed)
    print(args)

    # Load dataset
    data = get_dataset(args.dataset_path, args.dataset, True)
    feat_dim = data.features.shape[1]
    cluster_num = np.max(data.labels) + 1
    n_nodes = data.features.shape[0]

    # Initialize encoder and discriminator
    encoder = VGAE(feat_dim, args.hid_dim[0], args.hid_dim[1], 0.0, cluster_num)
    modeldis = Discriminator(args.hid_dim[0], args.hid_dim[1], args.hid_dim[2])

    # Set SSL set
    # set_of_ssl = ['PairwiseAttrSim', 'PairwiseDistance', 'Par', 'Clu', 'DGI']
    set_of_ssl = ['Par']
    if data.adj.shape[0] > 5000:
        print("Using DGISample")
        local_set_of_ssl = [ssl if ssl != 'DGI' else 'DGISample' for ssl in set_of_ssl]
    else:
        local_set_of_ssl = set_of_ssl

    # Initialize MoeSSL model
    device = args.device
    model = MoeSSL(data, encoder, modeldis, local_set_of_ssl, args,device)

    # Define loss function and optimizer
    loss_fn = re_loss_func  # Use reconstruction loss for stage one
    loss_func = DyFSSLoss(model, loss_fn)

    # Stage one: Pretraining
    if args.use_ckpt == 1:
        print("Loading the SSL trained ARVGA model")
        save_path = f'./examples/dyfss/pretrained/TotalSSL_{args.encoder}_{args.dataset}_{args.hid_dim[0]}.npz'
        try:
            tlx.files.load_and_assign_npz(save_path, model.encoder)
        except Exception as e:
            print(f"TensorLayerX loading failed, trying other methods. Error: {e}")
    else:
        # Pretraining stage
        model.set_train()

        # Prepare pretraining data
        pretrain_data = {
            'stage': 'pretrain',
            'features': model.processed_data.features,
            'adj_norm': model.processed_data.adj_norm,
            'adj_label': model.processed_data.adj_label,
            'pos_weight': model.processed_data.pos_weight,
            'norm': model.processed_data.norm,
            'w_ssl': args.w_ssl_stage_one
        }

        # Pretraining optimizer
        optimizer_pretrain = tlx.optimizers.Adam(lr=args.lr_pretrain, weight_decay=0.0)
        # Pretraining weights
        train_weights = model.params
        # Pretraining trainer
        train_one_step_pretrain = TrainOneStep(loss_func, optimizer_pretrain, train_weights)

        # Pretraining loop
        for epoch in range(args.pretrain_epochs):
            # Training
            train_loss = train_one_step_pretrain(pretrain_data, None)
            print(f'Epoch {epoch}, pretrain loss: {train_loss.item()}')

        # Save pretrained model
        if args.save_ckpt == 1:
            save_path = f"./examples/dyfss/pretrained/TotalSSL_{args.encoder}_{args.dataset}_{args.hid_dim[0]}.npz"
            tlx.files.save_npz(model.encoder.all_weights, save_path)

    # Get embeddings
    model.encoder.set_eval()
    bottom_embeddings, _, mu_z, _, _ = model.encoder(model.processed_data.features, model.processed_data.adj_norm)

    # Evaluate embeddings
    acc, nmi, f1, mu_pred, kmeans = evaluate_clustering(model, mu_z, data.labels, cluster_num)
    print('Z-embeddings clustering result: {:.2f}  {:.2f}  {:.2f}'.format(acc * 100, nmi * 100, f1 * 100))

    z_embeddings = tlx.ops.convert_to_tensor(tlx.ops.convert_to_numpy(mu_z))

    # Get pseudo labels
    cluster_center_z = tlx.convert_to_tensor(kmeans.cluster_centers_)
    selected_idx_z, pred_labels_z = generate_pseudo_labels_percentile(z_embeddings, cluster_center_z,
                                                                     data, cluster_num, args.st_per_p)

    # Stage two: Training
    print("Starting stage two training...")

    # Collect parameters to train
    train_weights = list(model.gate.trainable_weights)
    for ix, agent in enumerate(model.ssl_agent):
        if agent.disc2 is not None:
            train_weights = train_weights + list(agent.disc2.trainable_weights)
        if hasattr(agent, 'gcn2'):
            train_weights = train_weights + list(agent.gcn2.trainable_weights)

    # Get adjacency labels
    adj_label = get_adj_label(data.adj)

    # Initialize best result variables
    best_acc = 0
    best_nmi = 0
    best_f1 = 0
    best_epoch = 0
    best_pred = None

    # Get fused embeddings for initial evaluation
    model.set_train()
    fusion_emb, _, _ = model.FeatureFusionForward(z_embeddings)

    # Initial evaluation
    acc, nmi, f1, moe_pred_labels, kmeans = evaluate_clustering(model, fusion_emb, data.labels, cluster_num)
    print('Initial fusion_emb cluster result: {:.2f}  {:.2f}  {:.2f}'.format(acc * 100, nmi * 100, f1 * 100))

    # Prepare alignment labels
    print("Preparing alignment labels...")
    # Convert to numpy array for processing
    pred_labels_z_np = tlx.convert_to_numpy(pred_labels_z)
    # Use numpy to create one-hot encoding
    pseudo_labels_onehot_np = np.zeros((n_nodes, cluster_num))
    for i in range(n_nodes):
        pseudo_labels_onehot_np[i, pred_labels_z_np[i]] = 1.0

    pseudo_labels_onehot = tlx.convert_to_tensor(pseudo_labels_onehot_np, dtype=tlx.float32)
    pseudo_labels_onehot_cpu = tlx.convert_to_numpy(pseudo_labels_onehot)


    if not isinstance(moe_pred_labels, np.ndarray):
        moe_pred_labels_cpu = tlx.convert_to_numpy(moe_pred_labels)
    else:
        moe_pred_labels_cpu = moe_pred_labels


    try:
        print("Performing cluster alignment...")
        align_labels_z_np = cluster_alignment_simple(data, moe_pred_labels_cpu, [pseudo_labels_onehot_cpu])[0]
        align_labels_z = tlx.convert_to_tensor(align_labels_z_np)
    except Exception as e:
        print(f"Error during cluster alignment: {e}")
        print("Using original pseudo labels as alignment labels...")
        align_labels_z = pseudo_labels_onehot


    # Ensure cluster centers are correct trainable variables
    cluster_centers_np = kmeans.cluster_centers_
    cluster_centers = tlx.nn.Parameter(
        tlx.convert_to_tensor(cluster_centers_np, dtype=tlx.float32),
        name="cluster_centers"
    )
    # Add variable to the list of training weights
    train_weights.append(cluster_centers)



    # Create stage two loss function, optimizer, trainer
    loss_func_train = DyFSSTrainLoss(model, args.w_pq, args.w_ssl_stage_two)
    optimizer_train = tlx.optimizers.Adam(lr=args.lr_train_fusion, weight_decay=0.0)
    train_one_step = TrainOneStep(loss_func_train, optimizer_train, train_weights)

    # Prepare stage two training data
    train_data = {
        'z_embeddings': z_embeddings,
        'adj_norm': model.processed_data.adj_norm,
        'cluster_centers': cluster_centers,
        'selected_idx': selected_idx_z,
        'align_labels': align_labels_z,
        'adj_label': adj_label
    }

    # Stage two training loop
    print("Starting stage two training loop...")
    for epoch in range(args.labels_epochs):
        print(f"Epoch {epoch}")
        model.set_train()

        try:
            # Training step
            train_loss = train_one_step(train_data, None)

            # Get loss components
            if 'loss_components' in train_data:
                loss_components = train_data['loss_components']
                # Handle item() method for different backends
                if BACKEND == 'torch':
                    print(
                        "loss pq: {:.4f}, sup_loss: {:.4f}, ssl_loss: {:.4f}, sim_loss: {:.4f}".format(
                            loss_components['losspq'].item(),
                            loss_components['sup_loss'].item(),
                            loss_components['ssl_loss'].item(),
                            loss_components['sim_loss'].item()
                        )
                    )
                else:
                    # For non-PyTorch backends, use tensor values directly
                    print(
                        "loss pq: {:.4f}, sup_loss: {:.4f}, ssl_loss: {:.4f}, sim_loss: {:.4f}".format(
                            float(tlx.convert_to_numpy(loss_components['losspq'])),
                            float(tlx.convert_to_numpy(loss_components['sup_loss'])),
                            float(tlx.convert_to_numpy(loss_components['ssl_loss'])),
                            float(tlx.convert_to_numpy(loss_components['sim_loss']))
                        )
                    )

            # Evaluation step
            model.set_eval()
            if 'fusion_emb' in train_data and 'q' in train_data:
                fusion_emb_eval = train_data['fusion_emb']
                q_eval = train_data['q']

                # Convert q_eval to numpy for prediction
                q_eval_np = tlx.convert_to_numpy(q_eval)
                pred_labels_eval = np.argmax(q_eval_np, axis=1)

                # Evaluate clustering
                acc, nmi, f1 = cluster_accuracy(pred_labels_eval, data.labels, cluster_num)
                print('Epoch {} - ACC: {:.2f}, NMI: {:.2f}, F1: {:.2f}'.format(epoch, acc * 100, nmi * 100, f1 * 100))

                # Update best results
                if acc > best_acc:
                    best_acc = acc
                    best_nmi = nmi
                    best_f1 = f1
                    best_epoch = epoch
                    best_pred = pred_labels_eval

                    # Save the best model checkpoint if needed
                    if args.save_ckpt == 1:
                         save_path_best = f"./examples/dyfss/trained/Best_DyFSS_{args.dataset}_{args.hid_dim[0]}.npz"
                         tlx.files.save_npz(model.all_weights, save_path_best)
                         print(f"Best model saved at epoch {epoch} to {save_path_best}")

            else:
                 print("Warning: 'fusion_emb' or 'q' not found in train_data for evaluation.")

        except Exception as e:
            print(f"Error during training epoch {epoch}: {e}")
            # Optionally break the loop or handle the error appropriately
            break

    print("Stage two training finished.")
    print('Best Epoch: {}, Best ACC: {:.2f}, Best NMI: {:.2f}, Best F1: {:.2f}'.format(best_epoch, best_acc * 100, best_nmi * 100, best_f1 * 100))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DyFSS Training')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name: cora, citeseer, pubmed, wiki')
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    parser.add_argument('--encoder', type=str, default='VGAE', help='Encoder type')
    parser.add_argument('--hid_dim', type=list, default=[256, 128, 16], help='Hidden dimensions for encoder and discriminator')
    parser.add_argument('--lr_pretrain', type=float, default=0.001, help='Learning rate for pretraining')
    parser.add_argument('--lr_train_fusion', type=float, default=0.001, help='Learning rate for fusion training')
    parser.add_argument('--pretrain_epochs', type=int, default=600, help='Number of pretraining epochs')
    parser.add_argument('--labels_epochs', type=int, default=200, help='Number of fusion training epochs')
    parser.add_argument('--w_ssl_stage_one', type=float, default=0.1, help='Weight for SSL loss in stage one')
    parser.add_argument('--w_ssl_stage_two', type=float, default=0.1, help='Weight for SSL loss in stage two')
    parser.add_argument('--w_pq', type=float, default=0.1, help='Weight for PQ loss')
    parser.add_argument('--st_per_p', type=float, default=0.8, help='Percentile for pseudo label selection')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_ckpt', type=int, default=0, help='Whether to load pretrained model (1 for yes, 0 for no)')
    parser.add_argument('--save_ckpt', type=int, default=1, help='Whether to save model checkpoints (1 for yes, 0 for no)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or gpu)') # Example: 'gpu:0'

    args = parser.parse_args()

    # Set device
    if args.device.startswith('gpu'):
        try:
            tlx.set_device(args.device)
            print(f"Using device: {args.device}")
        except Exception as e:
            print(f"Failed to set device {args.device}, falling back to CPU. Error: {e}")
            tlx.set_device('cpu')
            args.device = 'cpu'
    else:
        tlx.set_device('cpu')
        print("Using device: CPU")


    main()

