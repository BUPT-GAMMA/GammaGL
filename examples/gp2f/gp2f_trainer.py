"""Memory-safe, source-faithful GP2F Cora-to-PubMed 1-shot reproduction.

The released implementation materializes several N x N matrices.  PubMed has
19,717 nodes, so this trainer evaluates exactly the same loss row chunks and
never creates an N x N dense adjacency or similarity matrix.
"""
import argparse
import copy
import os
import random

import numpy as np
import tensorlayerx as tlx
import torch
import torch.nn.functional as F

from gammagl.datasets import Planetoid
from gammagl.models.gp2f import (GP2FEncoder, GP2FFeatureProjector, GP2FPrompt,
                            GP2FDualBranch, GP2FClassifier)
from gammagl.utils import add_self_loops, calc_gcn_norm, remove_self_loops


def set_seed(seed):
    seed = int(seed)%(2**32-1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def row_normalize(x):
    return x / torch.clamp(x.sum(dim=1, keepdim=True), min=1e-12)


def normalized_edges(edge_index, num_nodes):
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    return edge_index, calc_gcn_norm(edge_index, num_nodes)


def create_split(labels, shot):
    """The official 1-shot split: 90% test, shot nodes/class from the rest."""
    num_nodes = labels.shape[0]
    permutation = torch.randperm(num_nodes, device=labels.device)
    test_idx, remaining = permutation[:int(0.9 * num_nodes)], permutation[int(0.9 * num_nodes):]
    remaining_labels = labels[remaining]
    train_idx = []
    for class_id in range(int(labels.max().item()) + 1):
        candidates = remaining[remaining_labels == class_id]
        if candidates.shape[0] < shot:
            raise RuntimeError("not enough nodes for class {}".format(class_id))
        train_idx.append(candidates[:shot])
    train_idx = torch.cat(train_idx)
    return train_idx[torch.randperm(train_idx.shape[0], device=labels.device)], test_idx


def normalise(x):
    return F.normalize(x, dim=-1)


def sparse_rows(edge_index, start, end, num_nodes, device):
    """Return binary target adjacency rows [start:end] without N x N storage."""
    rows = torch.zeros((end - start, num_nodes), dtype=torch.float32, device=device)
    src, dst = edge_index[0], edge_index[1]
    selected = (src >= start) & (src < end)
    rows[src[selected] - start, dst[selected]] = 1.0
    return rows


def edge_positive_sum(similarity_rows, edge_index, start, end, include_self):
    """Sum values at binary adjacency positives for every chunk row."""
    src, dst = edge_index[0], edge_index[1]
    selected = (src >= start) & (src < end)
    if not include_self:
        selected = selected & (src != dst)
    local_src, selected_dst = src[selected] - start, dst[selected]
    values = similarity_rows[local_src, selected_dst]
    result = torch.zeros(end - start, dtype=similarity_rows.dtype,
                         device=similarity_rows.device)
    return result.index_add(0, local_src, values)


def fused_similarity_rows(frozen_norm, prompted_norm, alpha, start, end):
    frozen = frozen_norm[start:end] @ frozen_norm.t()
    prompted = prompted_norm[start:end] @ prompted_norm.t()
    return alpha * frozen + (1.0 - alpha) * prompted, frozen, prompted


def topology_threshold(frozen_norm, prompted_norm, alpha, num_nodes, percentile):
    """Same 5,000 random-pair quantile as the released source."""
    sample_size = min(5000, num_nodes * (num_nodes - 1) // 2)
    row = torch.randint(num_nodes, (sample_size,), device=frozen_norm.device)
    col = torch.randint(num_nodes, (sample_size,), device=frozen_norm.device)
    valid = row != col
    values = alpha.detach() * (frozen_norm[row[valid]] * frozen_norm[col[valid]]).sum(dim=1)
    values += (1.0 - alpha.detach()) * (prompted_norm[row[valid]] * prompted_norm[col[valid]]).sum(dim=1)
    return torch.quantile(values, percentile / 100.0)


def chunk_loss(frozen_norm, prompted_norm, fusion_alpha, edge_index, num_nodes,
               start, end, temp_ctr, temp_topo, topo_threshold_value):
    fused, frozen, prompted = fused_similarity_rows(
        frozen_norm, prompted_norm, fusion_alpha, start, end)
    adjacency = sparse_rows(edge_index, start, end, num_nodes, fused.device)

    frozen_refl = torch.exp(frozen / temp_ctr)
    frozen_between = torch.exp((frozen_norm[start:end] @ prompted_norm.t()) / temp_ctr)
    prompted_refl = torch.exp(prompted / temp_ctr)
    prompted_between = torch.exp((prompted_norm[start:end] @ frozen_norm.t()) / temp_ctr)
    ctr1 = -torch.log((
        edge_positive_sum(frozen_refl, edge_index, start, end, False) +
        edge_positive_sum(frozen_between, edge_index, start, end, True)) /
        (frozen_refl.sum(dim=1) + frozen_between.sum(dim=1) -
         frozen_refl[torch.arange(end - start, device=fused.device),
                     torch.arange(start, end, device=fused.device)] + 1e-12))
    ctr2 = -torch.log((
        edge_positive_sum(prompted_refl, edge_index, start, end, False) +
        edge_positive_sum(prompted_between, edge_index, start, end, True)) /
        (prompted_refl.sum(dim=1) + prompted_between.sum(dim=1) -
         prompted_refl[torch.arange(end - start, device=fused.device),
                       torch.arange(start, end, device=fused.device)] + 1e-12))

    mask = ((fused > topo_threshold_value) & (adjacency > 0)) | \
           ((fused <= topo_threshold_value) & (adjacency <= 0))
    topo = F.binary_cross_entropy(torch.sigmoid(fused / temp_topo),
                                   adjacency * mask.float(), reduction='none')
    return (ctr1.sum() + ctr2.sum()) * 0.5, (topo * mask.float()).sum(), mask.sum()


def run_once(args, graph, cora_dim, seed, run):
    set_seed(seed * 1000 + run)
    x, labels = row_normalize(graph.x), graph.y.long()
    edge_index, _ = normalized_edges(graph.edge_index, graph.num_nodes)
    train_idx, test_idx = create_split(labels, args.shot)
    num_classes = int(labels.max().item()) + 1

    encoder = GP2FEncoder(cora_dim, args.hidden_dim, args.num_layers)
    encoder.load_weights(args.pretrained_path, format='npz_dict')
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.set_eval()

    projector = GP2FFeatureProjector(x.shape[1], cora_dim)
    prompt = GP2FPrompt(args.hidden_dim, args.num_layers, args.bottleneck_dim)
    fusion = GP2FDualBranch()
    classifier = GP2FClassifier(args.hidden_dim, num_classes)
    modules = [projector, prompt, fusion, classifier]
    optimizer = torch.optim.Adam(
        [weight for module in modules for weight in module.trainable_weights],
        lr=args.lr, weight_decay=args.weight_decay)
    best_loss, best_state, wait = float('inf'), None, 0

    for epoch in range(1, args.n_epoch + 1):
        for module in modules:
            module.set_train()
        optimizer.zero_grad()
        features = projector(x)
        frozen_output = encoder(features, edge_index, None, graph.num_nodes)
        prompted_output = prompt(features, edge_index, encoder, None, graph.num_nodes)
        frozen_norm, prompted_norm = normalise(frozen_output), normalise(prompted_output)

        threshold = topology_threshold(frozen_norm.detach(), prompted_norm.detach(),
                                       fusion.alpha, graph.num_nodes, args.topo_percentile)
        logits = classifier(fusion(frozen_output, prompted_output))
        cls = F.cross_entropy(logits[train_idx], labels[train_idx])
        
        cls.backward(retain_graph=True)

        frozen_leaf = frozen_norm.detach().requires_grad_(True)
        prompted_leaf = prompted_norm.detach().requires_grad_(True)

        ctr_total, topo_value = 0.0, 0.0
        starts = list(range(0, graph.num_nodes, args.sim_chunk_size))

        topo_denominator = 0
        with torch.no_grad():
            for start in starts:
                end = min(start + args.sim_chunk_size, graph.num_nodes)
                fused, _, _ = fused_similarity_rows(frozen_leaf, prompted_leaf,
                                                     fusion.alpha, start, end)
                adjacency = sparse_rows(edge_index, start, end, graph.num_nodes, x.device)
                mask = ((fused > threshold) & (adjacency > 0)) | ((fused <= threshold) & (adjacency <= 0))
                topo_denominator += int(mask.sum().cpu())

        for start in starts:
            end = min(start + args.sim_chunk_size, graph.num_nodes)
            ctr_sum, topo_sum, _ = chunk_loss(
                frozen_leaf, prompted_leaf, fusion.alpha, edge_index,
                graph.num_nodes, start, end, args.temp_ctr, args.temp_topo, threshold)
            (args.lambda_ctr * ctr_sum / graph.num_nodes +
             args.lambda_topo * topo_sum / topo_denominator).backward()
            ctr_total += float(ctr_sum.detach().cpu())
            topo_value += float(topo_sum.detach().cpu())

        torch.autograd.backward([frozen_norm, prompted_norm],
                                [frozen_leaf.grad, prompted_leaf.grad])
        optimizer.step()

        value = float(cls.detach().cpu()) + args.lambda_ctr * ctr_total / graph.num_nodes + \
            args.lambda_topo * topo_value / topo_denominator
        if value < best_loss:
            best_loss, wait = value, 0
            best_state = [copy.deepcopy(module.state_dict()) for module in modules]
        else:
            wait += 1
        if epoch % args.print_every == 0:
            print('seed={} run={} epoch={} loss={:.4f}'.format(seed, run, epoch, value))
        if wait >= args.patience:
            break

    for module, state in zip(modules, best_state):
        module.load_state_dict(state)
        module.set_eval()
    with torch.no_grad():
        features = projector(x)
        frozen_output = encoder(features, edge_index, None, graph.num_nodes)
        prompted_output = prompt(features, edge_index, encoder, None, graph.num_nodes)
        output = fusion(frozen_output, prompted_output)
        prediction = torch.argmax(classifier(output), dim=1)
        accuracy = (prediction[test_idx] == labels[test_idx]).float().mean().item() * 100.0
    print('method=gp2f seed={} run={} test_acc={:.2f}'.format(seed, run, accuracy))
    return accuracy


def main(args):
    if tlx.BACKEND != 'torch':
        raise RuntimeError('GP2F reproduction currently requires TL_BACKEND=torch')
    if not os.path.isfile(args.pretrained_path):
        raise FileNotFoundError('pretrained checkpoint not found: {}'.format(args.pretrained_path))

    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")


    pubmed, cora = Planetoid(args.dataset_path, 'pubmed'), Planetoid(args.dataset_path, 'cora')
    scores = []
    for seed in [int(value) for value in args.seeds.split(',') if value.strip()]:
        for run in range(args.runs):
            scores.append(run_once(args, pubmed[0], cora.num_node_features, seed, run))
    scores = np.asarray(scores)
    print('GP2F Cora->PubMed 1-shot: {:.2f} +/- {:.2f} ({} runs)'.format(
        scores.mean(), scores.std(), len(scores)))
    if args.result_path:
        result_dir = os.path.dirname(args.result_path)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        np.save(args.result_path, scores)
        print('Saved {} per-run scores to {}'.format(len(scores), args.result_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--pretrained_path', type=str, default='output/gp2f/grace_cora_gp2f_128.npz')
    parser.add_argument('--result_path', type=str, default='')
    parser.add_argument('--seeds', type=str, default='12345,23456,34567,45678,56789')
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--bottleneck_dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--temp_ctr', type=float, default=0.2)
    parser.add_argument('--temp_topo', type=float, default=0.1)
    parser.add_argument('--topo_percentile', type=float, default=70.0)
    parser.add_argument('--lambda_topo', type=float, default=0.1)
    parser.add_argument('--lambda_ctr', type=float, default=0.05)
    parser.add_argument('--sim_chunk_size', type=int, default=4096)
    parser.add_argument('--print_every', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)
    main(parser.parse_args())

