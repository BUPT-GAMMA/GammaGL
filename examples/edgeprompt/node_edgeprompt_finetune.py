import argparse
import os
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.loader import DataLoader
from gammagl.models import EdgePromptGCNModel, EdgePromptNodeClassifier
from gammagl.utils import get_few_shot_split, node_subgraph
from tensorlayerx.model import TrainOneStep, WithLoss


HIDDEN_DIM = 128
NUM_LAYERS = 2
DROP_RATE = 0.5
NUM_ANCHORS = 10
BATCH_SIZE = 32
LR = 0.001
WEIGHT_DECAY = 0.0
NUM_HOPS = 2
TEST_RATIO = 0.2
DATA_ROOT = "Planetoid"


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tlx.set_seed(seed)


class NodeClsLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super().__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        del y
        logits = forward_subgraph_batch(self.backbone_network, data)
        return self._loss_fn(logits, tlx.reshape(data.y, (-1,)))


def forward_subgraph_batch(model, graph_batch):
    graph_emb = model.backbone(
        graph_batch,
        prompt_type=model.prompt_type,
        prompt=model.prompt,
        pooling="mean",
    )
    return model.classifier(graph_emb)


def evaluate(model, loader):
    total_loss = []
    pred_list = []
    label_list = []

    for batch in loader:
        logits = forward_subgraph_batch(model, batch)
        labels = tlx.reshape(batch.y, (-1,))
        loss = tlx.losses.softmax_cross_entropy_with_logits(logits, labels)
        total_loss.append(float(loss.item()))
        pred_list.extend(tlx.convert_to_numpy(tlx.argmax(logits, axis=-1)).tolist())
        label_list.extend(tlx.convert_to_numpy(labels).tolist())

    test_acc = float((np.asarray(pred_list) == np.asarray(label_list)).mean())
    test_loss = float(np.mean(total_loss))
    return test_loss, test_acc


def build_subgraphs(graph, labels, node_indices):
    labels_np = tlx.convert_to_numpy(tlx.reshape(labels, (-1,)))
    node_indices = tlx.convert_to_numpy(tlx.reshape(node_indices, (-1,))).astype(np.int64)
    graph_list = []
    for node_idx in node_indices.tolist():
        subgraph = node_subgraph(graph, int(node_idx), num_hops=NUM_HOPS)
        subgraph.y = tlx.convert_to_tensor(
            np.asarray([labels_np[int(node_idx)]], dtype=np.int64),
            dtype=tlx.int64,
        )
        graph_list.append(subgraph)
    return graph_list


def main(args):
    if not os.path.exists(args.pretrained_path):
        raise FileNotFoundError("Pretrained checkpoint not found: {}".format(args.pretrained_path))

    set_random_seed(args.seed)
    dataset = Planetoid(DATA_ROOT, args.dataset)
    graph = dataset[0]
    labels = tlx.reshape(graph.y, (-1,))
    train_idx, test_idx = get_few_shot_split(
        labels,
        num_shots=args.num_shots,
        test_ratio=TEST_RATIO,
        random_state=args.seed,
    )
    train_graphs = build_subgraphs(graph, labels, train_idx)
    test_graphs = build_subgraphs(graph, labels, test_idx)

    print(
        "Few-shot subgraphs on {}: train={}, test={} ({}-hop, test_ratio={:.2f})".format(
            args.dataset,
            len(train_graphs),
            len(test_graphs),
            NUM_HOPS,
            TEST_RATIO,
        )
    )

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    backbone = EdgePromptGCNModel(
        feature_dim=dataset.num_node_features,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        drop_rate=DROP_RATE,
        name="EdgePromptGCN",
    )
    backbone.load_weights(args.pretrained_path, format="npz_dict")

    model = EdgePromptNodeClassifier(
        backbone=backbone,
        num_classes=dataset.num_classes,
        prompt_type=args.method,
        num_prompts=NUM_ANCHORS,
        name="EdgePromptNodeClassifier",
    )

    optimizer = tlx.optimizers.Adam(lr=LR, weight_decay=WEIGHT_DECAY)
    loss_func = NodeClsLoss(model, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, model.tuning_weights())

    last_test_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.set_train()
        total_train_loss = []
        for batch in train_loader:
            train_loss = train_one_step(batch, batch.y)
            total_train_loss.append(float(train_loss.item()))
        train_loss = float(np.mean(total_train_loss))

        model.set_eval()
        test_loss, test_acc = evaluate(model, test_loader)
        last_test_acc = test_acc
        print(
            "Epoch [{:0>3d}]  train loss: {:.4f}  test loss: {:.4f}  test acc: {:.4f}".format(
                epoch,
                train_loss,
                test_loss,
                test_acc,
            )
        )

    print("Final test acc: {:.4f}".format(last_test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downstream task: node classification")
    parser.add_argument("--dataset", type=str, default="Cora", choices=["Cora", "CiteSeer", "PubMed"])
    parser.add_argument("--method", type=str, default="edgeprompt", choices=["edgeprompt", "edgeprompt_plus"])
    parser.add_argument("--num_shots", type=int, default=5)
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if tlx.BACKEND == "torch":
        try:
            import torch
            if torch.cuda.is_available():
                tlx.set_device("GPU", 0)
            else:
                tlx.set_device("CPU")
        except ImportError:
            tlx.set_device("CPU")
    else:
        tlx.set_device("CPU")

    main(args)
