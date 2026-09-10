# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import math
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.datasets import ZINC
from gammagl.loader import DataLoader
from gammagl.models import GNNPPModel


def get_rw_landing_probs_and_edge_features(ksteps, edge_index, num_nodes):
    source = edge_index[0]
    dest = edge_index[1]
    num_edges = source.shape[0]

    ones = tlx.ones((num_edges,), dtype=tlx.float32)
    degree = tlx.unsorted_segment_sum(
        ones,
        source,
        num_nodes
    )
    degree = tlx.where(
        degree > 0,
        degree,
        tlx.ones_like(degree)
    )
    edge_weight = 1.0 / tlx.gather(degree, source)
    flat_indices = source * num_nodes + dest
    flat_A = tlx.unsorted_segment_sum(
        edge_weight,
        flat_indices,
        num_nodes * num_nodes
    )
    P = tlx.reshape(
        flat_A,
        (num_nodes, num_nodes)
    )
    rw_node_list = []
    rw_edge_list = []
    P_power = tlx.eye(
        num_nodes,
        dtype=tlx.float32
    )
    edge_index_transpose = tlx.transpose(edge_index)
    for k in ksteps:
        P_power = tlx.matmul(P_power, P)
        rw_node = tlx.diag(P_power)
        rw_edge = tlx.gather_nd(
            P_power,
            edge_index_transpose
        )
        rw_node_list.append(
            tlx.expand_dims(rw_node, axis=1)
        )
        rw_edge_list.append(
            tlx.expand_dims(rw_edge, axis=1)
        )
    rw_node = tlx.concat(rw_node_list, axis=1)
    rw_edge = tlx.concat(rw_edge_list, axis=1)
    return rw_node, rw_edge

def add_rwse_to_dataset(dataset):
    data_list = []
    ksteps = list(range(1, 21))
    for i in range(len(dataset)):
        data = dataset[i]
        rw_node, edge_features = get_rw_landing_probs_and_edge_features(
            ksteps,
            data.edge_index,
            data.x.shape[0]
        )
        data.pestat_RWSE = rw_node
        edge_attr = data.edge_attr
        if len(edge_attr.shape) == 1:
            edge_attr = tlx.expand_dims(edge_attr, axis=1)
        data.edge_attr = tlx.concat(
            [edge_attr, edge_features],
            axis=1
        )
        data_list.append(data)
    return data_list

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(
            backbone=net,
            loss_fn=loss_fn
        )
    def forward(self, batch, y):
        pred = self.backbone_network(batch)
        y = tlx.reshape(y, pred.shape)
        loss = self._loss_fn(pred, y)
        return loss

def clip_grad_norm(weights, max_norm=1.0):
    total_norm = 0.0
    for weight in weights:
        if weight.grad is not None:
            grad = weight.grad
            total_norm += float(
                tlx.reduce_sum(grad * grad)
            )
    total_norm = total_norm ** 0.5
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for weight in weights:
            if weight.grad is not None:
                weight.grad *= scale

def main(args):
    os.makedirs(args.best_model_path, exist_ok=True)
    root_dir = args.dataset_path
    train_dataset = ZINC(
        root=root_dir,
        subset=True,
        split='train'
    )
    val_dataset = ZINC(
        root=root_dir,
        subset=True,
        split='val'
    )
    train_dataset = add_rwse_to_dataset(train_dataset)

    val_dataset = add_rwse_to_dataset(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    net = GNNPPModel(
        num_node_types=21,
        num_edge_types=4,
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        residual=True,
        ffn=True
    )

    net_with_loss = SemiSpvzLoss(
        net=net,
        loss_fn=tlx.losses.absolute_difference_error
    )

    optimizer = tlx.optimizers.Adam(
        lr=args.lr,
        weight_decay=1e-5,
        grad_clip=lambda weights: clip_grad_norm(
            weights,
            max_norm=1.0
        )
    )

    total_epochs = args.n_epoch
    warmup_epochs = 50

    train_one_step = TrainOneStep(
        net_with_loss=net_with_loss,
        optimizer=optimizer,
        train_weights=net.trainable_weights
    )

    best_val_mae = float("inf")

    for epoch in range(1, total_epochs + 1):
        if epoch <= warmup_epochs:
            lr = args.lr * epoch / warmup_epochs
        else:
            progress = (
                (epoch - warmup_epochs) /
                (total_epochs - warmup_epochs)
            )
            min_lr = 1e-6
            lr = min_lr + (
                args.lr - min_lr
            ) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        optimizer.lr = lr
        net.set_train()
        total_loss = 0

        for batch in train_loader:
            loss = train_one_step(batch, batch.y)
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f}"
        )
        print(
            f"Epoch {epoch:03d} | "
            f"LR: {lr:.8f}"
        )

        net.set_eval()

        val_abs_error = 0.0
        val_count = 0

        for batch in val_loader:
            pred = net(batch)
            true = tlx.reshape(batch.y, pred.shape)

            abs_error = tlx.abs(pred - true)
            val_abs_error += float(
                tlx.reduce_sum(abs_error)
            )
            val_count += abs_error.numel()
        val_mae = val_abs_error / val_count

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_path = os.path.join(
                args.best_model_path,
                "zinc_gnnpp_best.npz"
            )
            net.save_weights(save_path)
            print(f"Best model saved: {save_path}")
        print(
            f"          | Val MAE: {val_mae:.4f}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data",
        help="path to the dataset"
    )
    parser.add_argument(
        "--best_model_path",
        type=str,
        default="./checkpoints/",
        help="path to save the best model"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate"
    )
    parser.add_argument(
        "--n_epoch",
        type=int,
        default=2000,
        help="number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size"
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=70,
        help="hidden dimension"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=9,
        help="number of layers"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        help="dropout rate"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="gpu id, -1 for CPU"
    )

    args = parser.parse_args()

    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    main(args)
