import os
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('TL_BACKEND', 'torch')
import argparse
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.optimizers import Adam
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.utils import to_undirected, remove_self_loops
from dataset import NodeDataset
from gammagl.models.gnrf import GNN
import gc


def load_best_args(args):
    import json
    dataset = args.dataset.lower()
    with open(os.path.join(EXAMPLE_DIR, "args.json"), 'r') as f:
        json_data = json.load(f)
    if dataset in json_data:
        for key, value in json_data[dataset].items():
            if not hasattr(args, key):
                setattr(args, key, value)
    return args

class SemiSpvzLoss(WithLoss):
    def __init__(self, backbone, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=backbone, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x'], data['edge_index'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(y, data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss

def evaluate_loss(model, data, y, mask_idx, loss_fn):
    model.set_eval()
    logits = model(data['x'], data['edge_index'])
    target_logits = tlx.gather(logits, mask_idx)
    target_y = tlx.gather(y, mask_idx)
    loss = loss_fn(target_logits, target_y)
    if hasattr(loss, 'item'):
        loss_val = loss.item()
    elif hasattr(loss, 'numpy'):
        loss_val = float(loss.numpy())
    else:
        loss_val = float(loss)
    return loss_val

def evaluate_acc(model, data, y, mask_idx):
    model.set_eval()
    logits = model(data['x'], data['edge_index'])
    target_logits = tlx.gather(logits, mask_idx)
    target_y = tlx.gather(y, mask_idx)
    pred_class = tlx.argmax(target_logits, axis=1)
    pred_np = tlx.convert_to_numpy(pred_class)
    true_np = tlx.convert_to_numpy(target_y)
    correct = (pred_np == true_np)
    acc = correct.mean()
    return acc

def main(args):
    tlx.set_device(args.device)
    data = NodeDataset(args.dataset)
    if (args.dataset in ["cornell", "wisconsin", "texas"]) and args.rewiring is not None:
        rewired = np.load(os.path.join(EXAMPLE_DIR, "Datasets", "rewiring.npz"))
        edge_index = rewired[f"{args.rewiring}_{args.dataset}"]
        edge_index = tlx.convert_to_tensor(edge_index, dtype=tlx.int64)
        edge_index = to_undirected(edge_index)
        edge_index = remove_self_loops(edge_index)[0]
        edge_index = tlx.to_device(edge_index, args.device)
    else:
        edge_index = tlx.to_device(data.edge_index, args.device)
    x = tlx.to_device(data.x, args.device)
    y = tlx.to_device(data.y, args.device)
    args.num_feat = data.nfeat
    args.num_class = data.nclass

    results = []
    for trial in range(int(args.trial)):
        tlx.set_seed(trial)
        np.random.seed(trial)
        train_idx, val_idx, test_idx = data.random_split(seed=trial, p_train=0.6, p_val=0.2)
        train_idx = tlx.convert_to_tensor(train_idx, dtype=tlx.int64)
        val_idx = tlx.convert_to_tensor(val_idx, dtype=tlx.int64)
        test_idx = tlx.convert_to_tensor(test_idx, dtype=tlx.int64)
        train_idx = tlx.to_device(train_idx, args.device)
        val_idx = tlx.to_device(val_idx, args.device)
        test_idx = tlx.to_device(test_idx, args.device)

        data_dict = {'x': x,'edge_index': edge_index,'train_idx': train_idx}
        model = GNN(args)
        if hasattr(model, 'to_device'):
            model.to_device(args.device)
        elif hasattr(model, 'to'):
            model.to(args.device)
        optimizer = Adam(lr=args.lr, weight_decay=args.weight_decay)
        train_weights = model.trainable_weights
        loss_func = SemiSpvzLoss(model, tlx.losses.softmax_cross_entropy_with_logits)
        train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

        best_val_loss = float('inf')
        final_test_acc = 0.0
        for epoch in range(int(args.epoch)):
            model.set_train()
            train_loss = train_one_step(data_dict, y)
            if hasattr(train_loss, 'item'):
                train_loss = train_loss.item()
            elif hasattr(train_loss, 'numpy'):
                train_loss = float(train_loss.numpy())
            else:
                train_loss = float(train_loss)
            val_loss = evaluate_loss(model, data_dict, y, val_idx, tlx.losses.softmax_cross_entropy_with_logits)
            test_acc = evaluate_acc(model, data_dict, y, test_idx)

            if args.verbose:
                print(f"[Epoch {epoch:3d}] Train Loss: {train_loss:.4f}", \
                        f"Valid Loss: {val_loss:.4f}, Test Acc: {test_acc:.4f}.")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                final_test_acc = test_acc

            gc.collect()

        if args.verbose:
            print(f"Best Test Acc: {final_test_acc:.4f}")
        results.append(final_test_acc)

    results_np = np.array(results)
    mean, std = np.mean(results_np), np.std(results_np)
    print(f"Mean: {mean:.4f}, Std: {std:.4f}", flush=True)
    return mean


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args, unknown_args = parser.parse_known_args()
    unknown_args_dict = {}
    for i in range(0, len(unknown_args), 2):
        key = unknown_args[i].lstrip('-')
        value = unknown_args[i+1] if i+1 < len(unknown_args) else None
        unknown_args_dict[key] = value
    for key, value in unknown_args_dict.items():
        setattr(args, key, value)
    args = load_best_args(args)
    main(args)
