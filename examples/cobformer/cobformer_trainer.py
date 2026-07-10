#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   cobformer_trainer.py
@Time    :   2025/09/04 2:33:00
@Author  :   lzd
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR

import argparse
import random
import numpy as np
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import CoBFormerModel
from gammagl.utils import add_self_loops, mask_to_index
from tensorlayerx.model import TrainOneStep, WithLoss


def set_seed(seed=123):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    # Set seed for tensorlayerx backend
    tlx.set_seed(seed)


class CoBFormerLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(CoBFormerLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        pred1, pred2 = self.backbone_network(data['x'], data['patch'], data['edge_index'])
        # For training, we use the combined loss function from the model
        loss = self.backbone_network.loss(pred1, pred2, data['y'], data['train_mask'])
        return loss


def calculate_f1(pred, label, num_classes, average='micro'):
    """
    Calculate F1 score
    Args:
        pred: predicted labels
        label: true labels
        num_classes: number of classes
        average: 'micro' or 'macro'

    Returns:
        f1 score
    """
    # Convert logits to predicted classes if needed
    if len(tlx.get_tensor_shape(pred)) > 1 and tlx.get_tensor_shape(pred)[-1] > 1:
        pred = tlx.argmax(pred, axis=-1)
    
    # Ensure pred and label are the same dtype
    if pred.dtype != label.dtype:
        label = tlx.cast(label, dtype=pred.dtype)
    
    # For micro F1, we need to flatten predictions and labels
    if average == 'micro':
        # Calculate micro F1 directly using tensor operations that avoid conversion issues
        eq = (pred == label)
        # Convert boolean tensor to float directly
        correct = tlx.reduce_sum(tlx.cast(tlx.convert_to_tensor(eq), tlx.float32))
        total = tlx.get_tensor_shape(pred)[0]
        # Add numerical stability with epsilon
        epsilon = 1e-8
        return correct / (float(total) + epsilon)
    else:
        # For macro F1, calculate per-class F1 and average
        # This is a simplified version - in practice you might want to use a more complete implementation
        f1_scores = []
        for i in range(num_classes):
            # True positives, false positives, false negatives for class i
            # Use direct comparison that returns tensor operations
            eq_pred = (pred == i)
            eq_label = (label == i)
            ne_label = (label != i)
            ne_pred = (pred != i)
            
            # Use element-wise operations that work with tensors
            tp = tlx.reduce_sum(tlx.cast(tlx.convert_to_tensor(eq_pred) & tlx.convert_to_tensor(eq_label), tlx.float32))
            fp = tlx.reduce_sum(tlx.cast(tlx.convert_to_tensor(eq_pred) & tlx.convert_to_tensor(ne_label), tlx.float32))
            fn = tlx.reduce_sum(tlx.cast(tlx.convert_to_tensor(ne_pred) & tlx.convert_to_tensor(eq_label), tlx.float32))
            
            # Precision and recall for class i with improved numerical stability
            epsilon = 1e-8
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            
            # F1 score for class i
            f1 = 2 * precision * recall / (precision + recall + epsilon)
            f1_scores.append(f1)
        
        # Return average F1 score
        return tlx.reduce_mean(tlx.stack(f1_scores))

def calculate_acc(logits, y, metrics):
    """
    Args:
        logits: node logits
        y: node labels
        metrics: tensorlayerx.metrics

    Returns:
        rst
    """
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


def metis_partition(graph, n_patches):
    """
    METIS partitioning that matches the original CoBFormer implementation.
    """
    import numpy as np
    try:
        import networkx as nx
        import metis
        
        num_nodes = graph.num_nodes
        
        # Handle case where n_patches is greater than num_nodes
        if num_nodes < n_patches:
            # Create a random partition when we have more patches than nodes
            membership = np.random.randint(0, n_patches, num_nodes)
        else:
            # Create NetworkX graph from edge_index
            # Move tensor to CPU first if it's on GPU
            if hasattr(graph.edge_index, 'cpu'):
                edge_index = graph.edge_index.cpu().numpy()
            else:
                edge_index = graph.edge_index.numpy()
            
            # Create NetworkX graph
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            
            # Add edges to the graph
            for i in range(edge_index.shape[1]):
                G.add_edge(edge_index[0, i], edge_index[1, i])
            
            # Use METIS to partition the graph
            cuts, membership = metis.part_graph(G, n_patches, recursive=True)
        
        # Convert membership to patch format like the original implementation
        patch = []
        max_patch_size = 0
        for i in range(n_patches):
            patch_nodes = np.where(membership == i)[0].tolist()
            patch.append(patch_nodes)
            max_patch_size = max(max_patch_size, len(patch_nodes))
        
        # Handle case where max_patch_size is 0 (all patches are empty)
        if max_patch_size == 0:
            # Create a simple partition where each node is in its own patch
            max_patch_size = 1
            for i in range(min(n_patches, num_nodes)):
                patch[i] = [i]
            # Fill remaining patches with the last node
            for i in range(num_nodes, n_patches):
                patch[i] = [num_nodes - 1]
        
        # Pad each patch to max_patch_size
        for i in range(len(patch)):
            if len(patch[i]) == 0:
                # If patch is empty, fill it with the last node
                patch[i] = [num_nodes - 1] * max_patch_size
            else:
                patch[i] += [num_nodes - 1] * (max_patch_size - len(patch[i]))
        
        # Convert to tensor format
        patch_tensors = []
        for p in patch:
            patch_tensor = tlx.convert_to_tensor(p, dtype=tlx.int32)
            patch_tensors.append(tlx.expand_dims(patch_tensor, axis=0))
        
        # Concatenate all patches
        if patch_tensors:
            patch_result = tlx.concat(patch_tensors, axis=0)
        else:
            # Create a default patch if something went wrong
            patch_result = tlx.convert_to_tensor([[num_nodes - 1]], dtype=tlx.int32)
        
        print(f"METIS patch tensor shape: {tlx.get_tensor_shape(patch_result)}")
        print(f"METIS patch tensor dtype: {patch_result.dtype}")
        return patch_result
        
    except ImportError:
        # Fallback to simple partitioning if METIS or NetworkX is not available
        print("METIS or NetworkX not available, using simple partitioning")
        import numpy as np
        
        num_nodes = graph.num_nodes
        # Simple partitioning for demonstration - in practice use actual METIS
        patch_size = num_nodes // n_patches
        patch = []
        
        for i in range(n_patches):
            start = i * patch_size
            end = min((i + 1) * patch_size, num_nodes)
            patch.append(list(range(start, end)))
        
        # Add remaining nodes to last patch
        if num_nodes % n_patches != 0:
            remaining = list(range(n_patches * patch_size, num_nodes))
            patch[-1].extend(remaining)
        
        # Convert to tensor format using tlx operations to avoid torch.tensor warnings
        max_patch_size = max(len(p) for p in patch) if patch else 0
        
        # Handle case where max_patch_size is 0 (all patches are empty)
        if max_patch_size == 0:
            # Create a simple partition where each node is in its own patch
            max_patch_size = 1
            for i in range(min(n_patches, num_nodes)):
                if i < len(patch):
                    patch[i] = [i]
                else:
                    patch.append([i])
            # Fill remaining patches with the last node
            for i in range(len(patch), n_patches):
                patch.append([num_nodes - 1])
            max_patch_size = 1
        
        # Pad each patch to max_patch_size
        for i in range(len(patch)):
            if len(patch[i]) == 0:
                # If patch is empty, fill it with the last node
                patch[i] = [num_nodes - 1] * max_patch_size
            else:
                patch[i] += [num_nodes - 1] * (max_patch_size - len(patch[i]))
        
        # Create patch tensor using tlx operations
        patch_tensors = []
        for p in patch:
            # Convert each patch to tensor
            patch_tensor = tlx.convert_to_tensor(p, dtype=tlx.int32)
            patch_tensors.append(tlx.expand_dims(patch_tensor, axis=0))
        
        # Concatenate all patches
        if patch_tensors:
            patch_result = tlx.concat(patch_tensors, axis=0)
        else:
            # Create a default patch if something went wrong
            patch_result = tlx.convert_to_tensor([[num_nodes - 1]], dtype=tlx.int32)
        
        print(f"Fallback patch tensor shape: {tlx.get_tensor_shape(patch_result)}")
        print(f"Fallback patch tensor dtype: {patch_result.dtype}")
        return patch_result


def main(args):
    # load datasets
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes, n_loops=args.self_loops)

    # for mindspore, it should be passed into node indices
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)

    # Create patches for CoBFormer
    print("Creating patches...")
    patch = metis_partition(graph, args.n_patches)
    print(f"Patch created with shape: {tlx.get_tensor_shape(patch)}")
    
    # build model
    net = CoBFormerModel(num_nodes=graph.num_nodes,
                         in_channels=dataset.num_node_features,
                         hidden_channels=args.hidden_dim,
                         out_channels=dataset.num_classes,
                         layers=args.layers,
                         n_head=args.heads,
                         gcn_layers=args.gcn_layers,
                         alpha=args.alpha,
                         tau=args.tau,
                         dropout1=args.drop_rate1,
                         dropout2=args.drop_rate2,
                         gcn_use_bn=args.gcn_use_bn,
                         use_patch_attn=args.use_patch_attn,
                         name="CoBFormer")

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = CoBFormerLoss(net, None)  # Loss is computed within the model
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": edge_index,
        "patch": patch,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
        "train_mask": graph.train_mask,
        "test_mask": graph.test_mask,
        "val_mask": graph.val_mask,
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, graph.y)
        net.set_eval()
        
        # Get predictions from both models
        pred1, pred2 = net(data['x'], data['patch'], data['edge_index'])
        
        # Evaluate first model (GCN)
        val_logits1 = tlx.gather(pred1, data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        num_classes = tlx.get_tensor_shape(data['y'])[-1] if len(tlx.get_tensor_shape(data['y'])) > 1 else int(tlx.reduce_max(data['y'])) + 1
        # Add numerical stability check
        try:
            val_micro_f1_1 = calculate_f1(val_logits1, val_y, num_classes, 'micro')
            val_macro_f1_1 = calculate_f1(val_logits1, val_y, num_classes, 'macro')
        except Exception as e:
            print(f"Warning: Error calculating F1 for GCN model: {e}")
            val_micro_f1_1 = 0.0
            val_macro_f1_1 = 0.0
        
        # Evaluate second model (Transformer)
        val_logits2 = tlx.gather(pred2, data['val_idx'])
        try:
            val_micro_f1_2 = calculate_f1(val_logits2, val_y, num_classes, 'micro')
            val_macro_f1_2 = calculate_f1(val_logits2, val_y, num_classes, 'macro')
        except Exception as e:
            print(f"Warning: Error calculating F1 for Transformer model: {e}")
            val_micro_f1_2 = 0.0
            val_macro_f1_2 = 0.0

        print("Epoch [{:0>3d}] ".format(epoch + 1)               + "  train loss: {:.4f}".format(train_loss.item())               + "  val micro_f1_1: {:.4f}".format(val_micro_f1_1)               + "  val macro_f1_1: {:.4f}".format(val_macro_f1_1)               + "  val micro_f1_2: {:.4f}".format(val_micro_f1_2)               + "  val macro_f1_2: {:.4f}".format(val_macro_f1_2))

        # save best model on evaluation set (using the better of the two models)
        val_acc = max(val_micro_f1_1, val_micro_f1_2)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

    net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict')
    net.set_eval()
    
    # Get final predictions
    pred1, pred2 = net(data['x'], data['patch'], data['edge_index'])
    
    # Test first model (GCN)
    test_logits1 = tlx.gather(pred1, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    num_classes = tlx.get_tensor_shape(data['y'])[-1] if len(tlx.get_tensor_shape(data['y'])) > 1 else int(tlx.reduce_max(data['y'])) + 1
    try:
        test_micro_f1_1 = calculate_f1(test_logits1, test_y, num_classes, 'micro')
        test_macro_f1_1 = calculate_f1(test_logits1, test_y, num_classes, 'macro')
    except Exception as e:
        print(f"Warning: Error calculating test F1 for GCN model: {e}")
        test_micro_f1_1 = 0.0
        test_macro_f1_1 = 0.0
    
    # Test second model (Transformer)
    test_logits2 = tlx.gather(pred2, data['test_idx'])
    try:
        test_micro_f1_2 = calculate_f1(test_logits2, test_y, num_classes, 'micro')
        test_macro_f1_2 = calculate_f1(test_logits2, test_y, num_classes, 'macro')
    except Exception as e:
        print(f"Warning: Error calculating test F1 for Transformer model: {e}")
        test_micro_f1_2 = 0.0
        test_macro_f1_2 = 0.0
    
    print("Test Micro-F1 (GCN):  {:.4f}".format(test_micro_f1_1))
    print("Test Macro-F1 (GCN):  {:.4f}".format(test_macro_f1_1))
    print("Test Micro-F1 (Transformer):  {:.4f}".format(test_micro_f1_2))
    print("Test Macro-F1 (Transformer):  {:.4f}".format(test_macro_f1_2))


if __name__ == "__main__":
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--n_epoch", type=int, default=500, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=64, help="dimension of hidden layers")
    parser.add_argument("--layers", type=int, default=2, help="number of transformer layers")
    parser.add_argument("--heads", type=int, default=1, help="number of attention heads")
    parser.add_argument("--gcn_layers", type=int, default=2, help="number of GCN layers")
    parser.add_argument("--alpha", type=float, default=0.7, help="loss balancing parameter")
    parser.add_argument("--tau", type=float, default=0.3, help="temperature parameter")
    parser.add_argument("--drop_rate1", type=float, default=0.5, help="dropout rate 1")
    parser.add_argument("--drop_rate2", type=float, default=0.1, help="dropout rate 2")
    parser.add_argument("--l2_coef", type=float, default=1e-3, help="l2 loss coefficient")
    parser.add_argument('--dataset', type=str, default='Pubmed', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    parser.add_argument("--n_patches", type=int, default=224, help="number of patches for partitioning")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--gcn_use_bn', action='store_true', help='gcn use batch norm')
    parser.add_argument('--use_patch_attn', action='store_true', help='transformer use patch attention')
    parser.add_argument("--seed", type=int, default=123, help="random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    main(args)