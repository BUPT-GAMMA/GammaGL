import argparse
import os
import random

import numpy as np
import tensorlayerx as tlx
import torch

from gammagl.datasets import Planetoid
from gammagl.models.gp2f import GP2FGRACE
from gammagl.utils import add_self_loops, calc_gcn_norm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def row_normalize(x):
    return x / torch.clamp(x.sum(dim=1, keepdim=True), min=1e-12)


def drop_feature(x, drop_rate):
    keep = (torch.rand(x.shape[1], device=x.device) >= drop_rate).to(x.dtype)
    return x * keep.unsqueeze(0)


def drop_edge(edge_index, edge_weight, drop_rate):
    keep = torch.rand(edge_index.shape[1], device=edge_index.device) >= drop_rate
    return edge_index[:, keep], edge_weight[keep]


def main(args):
    if tlx.BACKEND != "torch":
        raise RuntimeError("GP2F reproduction currently requires TL_BACKEND=torch")

    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    set_seed(args.seed)
    dataset = Planetoid(args.dataset_path, "cora")
    graph = dataset[0]
    x = row_normalize(graph.x)
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
    edge_weight = calc_gcn_norm(edge_index, graph.num_nodes)

    model = GP2FGRACE(dataset.num_node_features, args.hidden_dim,
                       args.num_layers, args.temp)
    optimizer = torch.optim.Adam(model.trainable_weights, lr=args.lr,
                                 weight_decay=args.weight_decay)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    best_loss, wait = float("inf"), 0
    for epoch in range(1, args.n_epoch + 1):
        model.set_train()
        edge1, weight1 = drop_edge(edge_index, edge_weight, args.drop_edge_rate)
        edge2, weight2 = drop_edge(edge_index, edge_weight, args.drop_edge_rate)
        z1 = model.encode(drop_feature(x, args.drop_feature_rate), edge1, weight1, graph.num_nodes)
        z2 = model.encode(drop_feature(x, args.drop_feature_rate), edge2, weight2, graph.num_nodes)
        loss = model.loss(z1, z2)
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss, wait = loss_value, 0
            model.encoder.save_weights(args.save_path, format="npz_dict")
        else:
            wait += 1
        if epoch == 1 or epoch % args.print_every == 0:
            print("epoch={} loss={:.6f} best={:.6f}".format(epoch, loss_value, best_loss))
        if wait >= args.patience:
            print("Early stopping at epoch {}".format(epoch))
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Saved pretrained GRACE model to {}".format(args.save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='output/gp2f/grace_cora_gp2f_128.npz')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--drop_edge_rate', type=float, default=0.2)
    parser.add_argument('--drop_feature_rate', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)
    main(parser.parse_args())

