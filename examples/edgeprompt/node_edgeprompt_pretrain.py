import argparse
import json
import os
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorlayerx as tlx
from gammagl.data import Graph
from gammagl.datasets import Planetoid
from gammagl.models import EdgePromptGCNModel
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorlayerx.model import TrainOneStep, WithLoss


DATASET_NAME_MAP = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}


def canonical_dataset_name(dataset):
    dataset_key = str(dataset).lower()
    if dataset_key not in DATASET_NAME_MAP:
        raise ValueError("Unknown dataset: {}".format(dataset))
    return DATASET_NAME_MAP[dataset_key]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tlx.set_seed(seed)


def ensure_dir(path):
    if not path:
        return
    directory = path
    if os.path.splitext(path)[1]:
        directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def build_checkpoint_path(save_dir, best_model_path, filename):
    if best_model_path:
        if os.path.splitext(best_model_path)[1]:
            ensure_dir(best_model_path)
            return best_model_path
        ensure_dir(best_model_path)
        return os.path.join(best_model_path, filename)
    ensure_dir(save_dir)
    return os.path.join(save_dir, filename)


def load_node_dataset(dataset, dataset_path):
    dataset_name = canonical_dataset_name(dataset)
    dataset_obj = Planetoid(os.path.join(dataset_path, "Planetoid"), dataset_name)
    return dataset_name, dataset_obj, dataset_obj[0]


def empty_edge_index():
    return tlx.convert_to_tensor(np.zeros((2, 0), dtype=np.int64), dtype=tlx.int64)


def gather_edge_pairs(edge_pairs, indices):
    if len(indices) == 0:
        return empty_edge_index()
    return tlx.gather(
        edge_pairs,
        tlx.convert_to_tensor(indices, dtype=tlx.int64),
        axis=1,
    )


def unique_undirected_edge_pairs(edge_index):
    edge_np = np.asarray(tlx.convert_to_numpy(edge_index), dtype=np.int64)
    if edge_np.size == 0:
        return empty_edge_index()

    row = edge_np[0]
    col = edge_np[1]
    mask = row != col
    row = row[mask]
    col = col[mask]
    if row.size == 0:
        return empty_edge_index()

    pairs = np.stack([np.minimum(row, col), np.maximum(row, col)], axis=1)
    pairs = np.unique(pairs, axis=0)
    return tlx.convert_to_tensor(pairs.T, dtype=tlx.int64)


def to_bidirectional_edge_index(edge_pairs):
    if int(edge_pairs.shape[1]) == 0:
        return empty_edge_index()
    reverse_pairs = tlx.stack([edge_pairs[1], edge_pairs[0]], axis=0)
    return tlx.concat([edge_pairs, reverse_pairs], axis=1)


def build_graph_from_edge_pairs(graph, edge_pairs):
    return Graph(
        x=graph.x,
        edge_index=to_bidirectional_edge_index(edge_pairs),
        y=graph.y,
        num_nodes=graph.num_nodes,
    )


def split_edge_pairs(edge_pairs, val_ratio, test_ratio, seed):
    num_edges = int(edge_pairs.shape[1])
    if num_edges == 0:
        raise ValueError("No edges are available for pretraining.")

    num_val = max(1, int(num_edges * val_ratio))
    num_test = max(1, int(num_edges * test_ratio))
    if num_val + num_test >= num_edges:
        raise ValueError("val_ratio + test_ratio leaves no training edges.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_edges)
    val_idx = perm[:num_val]
    test_idx = perm[num_val:num_val + num_test]
    train_idx = perm[num_val + num_test:]
    return (
        gather_edge_pairs(edge_pairs, train_idx),
        gather_edge_pairs(edge_pairs, val_idx),
        gather_edge_pairs(edge_pairs, test_idx),
    )


def sample_negative_edge_pairs(num_nodes, positive_edge_pairs, num_samples, seed):
    if num_samples <= 0:
        return empty_edge_index()

    positive_np = tlx.convert_to_numpy(positive_edge_pairs).T
    positive_set = {(int(u), int(v)) for u, v in positive_np}
    negative_pairs = []
    negative_set = set()
    rng = np.random.default_rng(seed)
    max_trials = max(num_samples * 50, 10000)
    trials = 0
    while len(negative_pairs) < num_samples and trials < max_trials:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        trials += 1
        if u == v:
            continue
        pair = (u, v) if u < v else (v, u)
        if pair in positive_set or pair in negative_set:
            continue
        negative_set.add(pair)
        negative_pairs.append(pair)

    if len(negative_pairs) < num_samples:
        raise ValueError("Unable to sample enough negative node pairs.")

    return tlx.convert_to_tensor(np.asarray(negative_pairs, dtype=np.int64).T, dtype=tlx.int64)


def prepare_edge_prediction_splits(graph, val_ratio, test_ratio, neg_ratio, seed):
    full_edge_pairs = unique_undirected_edge_pairs(graph.edge_index)
    train_edge_pairs, val_edge_pairs, test_edge_pairs = split_edge_pairs(
        full_edge_pairs,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_message_graph = build_graph_from_edge_pairs(graph, train_edge_pairs)
    num_val_neg = max(1, int(val_edge_pairs.shape[1] * neg_ratio))
    num_test_neg = max(1, int(test_edge_pairs.shape[1] * neg_ratio))
    val_neg_pairs = sample_negative_edge_pairs(graph.num_nodes, full_edge_pairs, num_val_neg, seed + 1)
    test_neg_pairs = sample_negative_edge_pairs(graph.num_nodes, full_edge_pairs, num_test_neg, seed + 2)

    return {
        "full_edge_pairs": full_edge_pairs,
        "train_edge_pairs": train_edge_pairs,
        "val": {
            "message_graph": train_message_graph,
            "pos_edge_index": val_edge_pairs,
            "neg_edge_index": val_neg_pairs,
        },
        "test": {
            "message_graph": train_message_graph,
            "pos_edge_index": test_edge_pairs,
            "neg_edge_index": test_neg_pairs,
        },
    }


def sample_masked_edge_prediction_task(graph, train_edge_pairs, full_edge_pairs, mask_ratio, neg_ratio, seed):
    num_edges = int(train_edge_pairs.shape[1])
    if num_edges == 0:
        raise ValueError("No training edges are available after the pretraining split.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_edges)
    num_masked = max(1, int(num_edges * mask_ratio))
    pos_idx = perm[:num_masked]
    keep_idx = perm[num_masked:]

    pos_edge_pairs = gather_edge_pairs(train_edge_pairs, pos_idx)
    message_edge_pairs = gather_edge_pairs(train_edge_pairs, keep_idx)
    num_neg_edges = max(1, int(pos_edge_pairs.shape[1] * neg_ratio))
    neg_edge_pairs = sample_negative_edge_pairs(graph.num_nodes, full_edge_pairs, num_neg_edges, seed)

    return {
        "message_graph": build_graph_from_edge_pairs(graph, message_edge_pairs),
        "pos_edge_index": pos_edge_pairs,
        "neg_edge_index": neg_edge_pairs,
    }


def edge_scores(node_emb, edge_index):
    src_x = tlx.gather(node_emb, edge_index[0])
    dst_x = tlx.gather(node_emb, edge_index[1])
    return tlx.reduce_sum(src_x * dst_x, axis=-1)


def link_prediction_loss(pos_logits, neg_logits):
    pos_loss = tlx.losses.sigmoid_cross_entropy(pos_logits, tlx.ones_like(pos_logits))
    neg_loss = tlx.losses.sigmoid_cross_entropy(neg_logits, tlx.zeros_like(neg_logits))
    return pos_loss + neg_loss


class EdgePredLoss(WithLoss):
    def __init__(self, net):
        super().__init__(backbone=net, loss_fn=None)

    def forward(self, data, label):
        del label
        node_emb = self.backbone_network(data["message_graph"])
        pos_logits = edge_scores(node_emb, data["pos_edge_index"])
        neg_logits = edge_scores(node_emb, data["neg_edge_index"])
        return link_prediction_loss(pos_logits, neg_logits)


def evaluate(model, data):
    node_emb = model(data["message_graph"])
    pos_logits = edge_scores(node_emb, data["pos_edge_index"])
    neg_logits = edge_scores(node_emb, data["neg_edge_index"])
    loss = link_prediction_loss(pos_logits, neg_logits)

    pos_scores = tlx.convert_to_numpy(tlx.sigmoid(pos_logits))
    neg_scores = tlx.convert_to_numpy(tlx.sigmoid(neg_logits))
    scores = np.concatenate([pos_scores, neg_scores], axis=0)
    labels = np.concatenate([
        np.ones(pos_scores.shape[0], dtype=np.int64),
        np.zeros(neg_scores.shape[0], dtype=np.int64),
    ], axis=0)
    return float(loss.item()), roc_auc_score(labels, scores), average_precision_score(labels, scores)


def save_metadata(checkpoint_path, args, dataset_name):
    metadata_path = os.path.splitext(checkpoint_path)[0] + ".json"
    metadata = {
        "dataset": dataset_name,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "drop_rate": args.drop_rate,
        "mask_ratio": args.mask_ratio,
        "neg_ratio": args.neg_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "checkpoint_contains": "EdgePromptGCNModel backbone weights only",
        "note": "Minimal EP-GPPT-style engineering pretraining, not an official standalone script.",
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main(args):
    set_random_seed(args.seed)
    dataset_name, dataset, graph = load_node_dataset(args.dataset, args.dataset_path)
    splits = prepare_edge_prediction_splits(
        graph,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )

    print(
        "Edge pretrain split on {}: train_message_edges={}, val_pos={}, test_pos={}".format(
            dataset_name,
            int(splits["train_edge_pairs"].shape[1]),
            int(splits["val"]["pos_edge_index"].shape[1]),
            int(splits["test"]["pos_edge_index"].shape[1]),
        )
    )

    backbone = EdgePromptGCNModel(
        feature_dim=dataset.num_node_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        drop_rate=args.drop_rate,
        name="EdgePromptGCN",
    )
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
    loss_func = EdgePredLoss(backbone)
    train_one_step = TrainOneStep(loss_func, optimizer, backbone.trainable_weights)

    checkpoint_path = build_checkpoint_path(
        args.save_dir,
        args.best_model_path,
        "{}_ep_gppt_backbone.npz".format(dataset_name.lower()),
    )

    best_val_ap = -1.0
    for epoch in range(args.epochs):
        backbone.set_train()
        train_task = sample_masked_edge_prediction_task(
            graph,
            splits["train_edge_pairs"],
            splits["full_edge_pairs"],
            mask_ratio=args.mask_ratio,
            neg_ratio=args.neg_ratio,
            seed=args.seed + epoch,
        )
        train_loss = train_one_step(train_task, tlx.convert_to_tensor([1]))

        backbone.set_eval()
        val_loss, val_auc, val_ap = evaluate(backbone, splits["val"])
        print(
            "Epoch [{:0>3d}]  train loss: {:.4f}  val loss: {:.4f}  val auc: {:.4f}  val ap: {:.4f}".format(
                epoch + 1,
                float(train_loss.item()),
                val_loss,
                val_auc,
                val_ap,
            )
        )

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            backbone.save_weights(checkpoint_path, format="npz_dict")

    backbone.load_weights(checkpoint_path, format="npz_dict")
    backbone.set_eval()
    test_loss, test_auc, test_ap = evaluate(backbone, splits["test"])
    save_metadata(checkpoint_path, args, dataset_name)
    print("Best checkpoint: {}".format(checkpoint_path))
    print("Test loss: {:.4f}".format(test_loss))
    print("Test auc: {:.4f}".format(test_auc))
    print("Test ap: {:.4f}".format(test_ap))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora", help="dataset")
    parser.add_argument("--dataset_path", type=str, default=r"", help="path to save dataset")
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="number of GCN layers")
    parser.add_argument("--drop_rate", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--mask_ratio", type=float, default=0.1, help="masked edge ratio")
    parser.add_argument("--neg_ratio", type=float, default=1.0, help="negative sampling ratio")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="validation edge ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="test edge ratio")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--epochs", type=int, default=200, help="training epochs")
    parser.add_argument("--save_dir", type=str, default=r"./", help="path to save checkpoints")
    parser.add_argument("--best_model_path", type=str, default="", help="optional checkpoint file path or checkpoint directory")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    main(args)
