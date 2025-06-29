import tensorlayerx as tlx
import argparse
import time

from gammagl.data import Graph
from gammagl.utils import mask_to_index
from gammagl.datasets import Planetoid
from gammagl.models import NASA_GCN
from gammagl.transforms import NR_Augmentor
from gammagl.utils import accuracy_tlx, compute_gcn_norm

def main(args):

    try:
        tlx.set_device(device='GPU', id=args.gpu_id)
        print(f"Using GPU: {args.gpu_id}")
    except:
        tlx.set_device(device='CPU')
        print("GPU not available, using CPU.")

    try:
        dataset = Planetoid(root=args.dataset_path, name=args.dataset)
    except Exception as e:
        print(f"Error loading dataset {args.dataset}: {e}")
        print("Please ensure the dataset name is correct (Cora, Citeseer, PubMed) and it's downloadable.")
        return

    graph = dataset[0]
    num_nodes = graph.num_nodes
    
    graph_x = tlx.convert_to_tensor(graph.x, dtype=tlx.float32)
    graph_y = tlx.convert_to_tensor(graph.y, dtype=tlx.int64)
    graph_train_mask = tlx.convert_to_tensor(graph.train_mask, dtype=tlx.bool)
    graph_val_mask = tlx.convert_to_tensor(graph.val_mask, dtype=tlx.bool)
    graph_test_mask = tlx.convert_to_tensor(graph.test_mask, dtype=tlx.bool)
    
    eval_edge_index, eval_edge_weight = compute_gcn_norm(
        graph.edge_index, num_nodes, dtype=graph_x.dtype, add_self_loops_flag=True
    )
    eval_edge_index = tlx.convert_to_tensor(eval_edge_index)
    eval_edge_weight = tlx.convert_to_tensor(eval_edge_weight)

    # initialize
    augmentor = NR_Augmentor(probability=args.nr_prob)
    
    model = NASA_GCN(
        feature_dim=dataset.num_node_features,
        hidden_dim=args.hidden_dim,
        num_classes=dataset.num_classes,
        dropout_rate=args.dropout,
        temp=args.temp,
        alpha=args.alpha
    )

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
    train_weights = model.trainable_weights

    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0

    print("Starting training...")
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.set_train()

        # 1. Dynamic Augmentation
        temp_original_graph_for_aug = Graph(x=graph_x, edge_index=graph.edge_index, num_nodes=num_nodes)
        augmented_graph = augmentor.augment(temp_original_graph_for_aug)
        
        # 2. Preprocess augmented graph for GCN
        aug_edge_index, aug_edge_weight = compute_gcn_norm(
            augmented_graph.edge_index, 
            tlx.get_tensor_shape(augmented_graph.x)[0],
            dtype=augmented_graph.x.dtype,
            add_self_loops_flag=True
        )
        aug_edge_index = tlx.convert_to_tensor(aug_edge_index)
        aug_edge_weight = tlx.convert_to_tensor(aug_edge_weight)

        import tensorflow as tf
        with tf.GradientTape() as tape:
            output_logits_aug = model(
                augmented_graph.x,
                aug_edge_index,
                edge_weight=aug_edge_weight,
                num_nodes=tlx.get_tensor_shape(augmented_graph.x)[0]
            )

            loss = model.compute_nasa_loss(
                output_logits_aug,
                augmented_graph,
                graph_y,
                graph_train_mask
            )
        
        gradients = tape.gradient(loss, train_weights)
        optimizer.apply_gradients(zip(gradients, train_weights))

        # evaluation
        model.set_eval()

        eval_logits = model.predict(
            graph_x,
            eval_edge_index,
            edge_weight=eval_edge_weight,
            num_nodes=num_nodes
        )
        eval_pred_softmax = tlx.softmax(eval_logits, axis=-1)

        train_acc = accuracy_tlx(tlx.gather(eval_pred_softmax, mask_to_index(graph_train_mask)), 
                                 tlx.gather(graph_y, mask_to_index(graph_train_mask)))
        val_acc = accuracy_tlx(tlx.gather(eval_pred_softmax, mask_to_index(graph_val_mask)),
                               tlx.gather(graph_y, mask_to_index(graph_val_mask)))
        test_acc = accuracy_tlx(tlx.gather(eval_pred_softmax, mask_to_index(graph_test_mask)),
                                tlx.gather(graph_y, mask_to_index(graph_test_mask)))

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1:03d}/{args.epochs} | Loss: {loss.item():.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | "
              f"Time: {epoch_duration:.2f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch +1
            # save the best model
            # model.save_weights(f"nasa_gcn_{args.dataset}_best.npz")
            # print(f"New best validation accuracy: {best_val_acc:.4f}, saving model.")

    print("Training finished.")
    print(f"Best Epoch: {best_epoch}, Best Val Acc: {best_val_acc:.4f}, Corresponding Test Acc: {best_test_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NASA GCN training with GammaGL")
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name (Cora, Citeseer, PubMed)')
    parser.add_argument('--dataset_path', type=str, default='./data', help='Path to store/load datasets')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for Adam optimizer')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Number of hidden units in GCN')
    parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate for GCN layers and input features')
    # NASA specific hyperparameters
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for L_CR loss component')
    parser.add_argument('--temp', type=float, default=0.5, help='Temperature for sharpening pseudo-labels')
    parser.add_argument('--nr_prob', type=float, default=0.5, help='Probability for Neighbor Replacement')
    
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use, -1 for CPU')
    
    cli_args = parser.parse_args()
    print("Arguments:", cli_args)
    main(cli_args)