import os

# os.environ['TL_BACKEND'] = 'torch'  # set your backend here, default `tensorflow`
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from gammagl.datasets.alircd import AliRCD
import argparse
import tensorlayerx as tlx
from gammagl.layers.conv import RGCNConv
from tensorlayerx.model import TrainOneStep, WithLoss

from gammagl.loader import Hetero_Neighbor_Sampler as NeighborLoader
import os.path as osp
import numpy as np

from sklearn.metrics import average_precision_score


class RGCN(tlx.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_bases, n_layers=2):
        super().__init__()
        self.convs = tlx.nn.ModuleList()
        # self.relu = F.relu
        self.convs.append(RGCNConv(in_channels=in_channels, out_channels=hidden_channels, num_relations=num_relations,
                                   num_bases=num_bases))
        for i in range(n_layers - 2):
            self.convs.append(
                RGCNConv(in_channels=hidden_channels, out_channels=hidden_channels, num_relations=num_relations,
                         num_bases=num_bases))
        self.convs.append(RGCNConv(in_channels=hidden_channels, out_channels=out_channels, num_relations=num_relations,
                                   num_bases=num_bases))

    def forward(self, x, edge_index, edge_type, drop_rate=0.4):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = tlx.relu(x)
                x = tlx.layers.Dropout(p=drop_rate)(x)
        return x


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self._backbone(tlx.to_device(data['x'], device="gpu", id=data["id"]),
                                tlx.to_device(data['edge_index'], device="gpu", id=data["id"]),
                                tlx.to_device(data['edge_type'], device="gpu", id=data["id"]))
        train_logits = tlx.gather(logits, tlx.arange(data['start'], data['end']))
        train_labels = tlx.to_device(label, device="gpu", id=data["id"])
        loss = self._loss_fn(train_logits, train_labels)
        return loss


def evaluate(net, data, y, metrics):
    net.set_eval()
    logits = net(data['edge_index'], data['edge_type'])
    _logits = tlx.gather(logits, data['test_idx'])
    _y = y  # tlx.gather(y, data['test_idx'])
    metrics.update(_logits, _y)
    acc = metrics.result()
    metrics.reset()
    return acc


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


def main(args):
    data = AliRCD(args.dataset)
    hgraph = data[0]
    if tlx.BACKEND == 'torch':
        import torch
        device = torch.device(args.device_id if torch.cuda.is_available() else 'cpu')
    labeled_class = args.labeled_class

    train_idx = hgraph[labeled_class].pop('train_idx')
    val_idx = hgraph[labeled_class].pop('val_idx')
    test_idx = hgraph[labeled_class].pop('test_idx')

    # Mini-Batch

    train_loader = NeighborLoader(graph=hgraph, input_nodes=(labeled_class, train_idx),
                                  num_neighbors=[args.fanout] * args.n_layers,
                                  shuffle=True, batch_size=args.batch_size)

    val_loader = NeighborLoader(graph=hgraph, input_nodes=(labeled_class, val_idx),
                                num_neighbors=[args.fanout] * args.n_layers,
                                shuffle=False, batch_size=args.batch_size)

    test_loader = NeighborLoader(graph=hgraph, input_nodes=(labeled_class, test_idx),
                                 num_neighbors=[args.fanout] * args.n_layers,
                                 shuffle=False, batch_size=args.batch_size)

    num_relations = len(hgraph.edge_types)

    if tlx.BACKEND == 'torch':
        net = RGCN(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2, n_layers=args.n_layers,
                   num_relations=num_relations, num_bases=args.n_bases).to(device)
    else:
        net = RGCN(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2, n_layers=args.n_layers,
                   num_relations=num_relations, num_bases=args.n_bases)
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)

    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    best_val_acc = 0

    if args.inference == False:
        for epoch in range(args.n_epoch):
            train_loss = 0
            train_loss_list = []
            val_loss = val_acc = val_ap = 0
            test_loss = test_acc = test_ap = 0
            val_loss_list = []
            val_scores_list = []
            val_labels_list = []
            test_loss_list = []
            test_scores_list = []
            test_labels_list = []


            metric = tlx.metrics.Accuracy()

            net.set_train()
            for batch in train_loader:
                start = 0
                for ntype in batch.node_types:
                    if ntype == labeled_class:
                        break
                    start += batch[ntype].num_nodes
                batch = batch.tensor()
                batch_size = batch[labeled_class].batch_size
                label = tlx.gather(batch[labeled_class].y, tlx.arange(0, batch_size))
                batch = batch.to_homogeneous()
                data = {
                    'x': batch.x,
                    'edge_index': batch.edge_index,
                    'edge_type': batch.edge_type,
                    'start': start,
                    'end': start + batch_size,
                    "id": args.device_id,
                }
                loss = train_one_step(data=data, label=label)
                train_loss_list.append(loss)

            train_loss = np.hstack(train_loss_list).mean()

            net.set_eval()
            for batch in val_loader:
                start = 0
                for ntype in batch.node_types:
                    if ntype == labeled_class:
                        break
                    start += batch[ntype].num_nodes
                batch = batch.tensor()
                batch_size = batch[labeled_class].batch_size
                label = tlx.gather(batch[labeled_class].y, tlx.arange(0, batch_size))
                batch = batch.to_homogeneous()
                data = {
                    'x': batch.x,
                    'edge_index': batch.edge_index,
                    'edge_type': batch.edge_type,
                    'start': start,
                    'end': start + batch_size,
                    "id": args.device_id,
                }
                logits = net(tlx.to_device(data["x"], device="gpu", id=args.device_id),
                             tlx.to_device(data["edge_index"], device="gpu", id=args.device_id),
                             tlx.to_device(data["edge_type"], device="gpu", id=args.device_id))
                val_logits = tlx.gather(logits, tlx.arange(data['start'], data['end']))
                val_labels = tlx.to_device(label, device="gpu", id=data["id"])
                loss = tlx.losses.softmax_cross_entropy_with_logits(val_logits, val_labels)
                val_loss_list.append(tlx.convert_to_numpy(loss))
                val_preds = tlx.nn.Softmax()(val_logits)
                metric.update(val_preds, val_labels)
                val_scores_list.append(tlx.convert_to_numpy(val_preds)[:, 1])
                val_labels_list.append(tlx.convert_to_numpy(val_labels))

            val_loss = np.hstack(val_loss_list).mean()
            val_acc = metric.result()
            val_ap = average_precision_score(np.hstack(val_labels_list), np.hstack(val_scores_list))

            for batch in test_loader:
                start = 0
                for ntype in batch.node_types:
                    if ntype == labeled_class:
                        break
                    start += batch[ntype].num_nodes
                batch = batch.tensor()
                batch_size = batch[labeled_class].batch_size
                label = tlx.gather(batch[labeled_class].y, tlx.arange(0, batch_size))
                batch = batch.to_homogeneous()
                data = {
                    'x': batch.x,
                    'edge_index': batch.edge_index,
                    'edge_type': batch.edge_type,
                    'start': start,
                    'end': start + batch_size,
                    "id": args.device_id,
                }
                logits = net(tlx.to_device(data["x"], device="gpu", id=args.device_id),
                             tlx.to_device(data["edge_index"], device="gpu", id=args.device_id),
                             tlx.to_device(data["edge_type"], device="gpu", id=args.device_id))
                test_logits = tlx.gather(logits, tlx.arange(data['start'], data['end']))
                test_labels = tlx.to_device(label, device="gpu", id=data["id"])
                loss = tlx.losses.softmax_cross_entropy_with_logits(test_logits, test_labels)
                test_loss_list.append(tlx.convert_to_numpy(loss))
                test_preds = tlx.nn.Softmax()(test_logits)
                metric.update(test_preds, test_labels)
                test_scores_list.append(tlx.convert_to_numpy(test_preds)[:, 1])
                test_labels_list.append(tlx.convert_to_numpy(test_labels))

            test_loss = np.hstack(val_loss_list).mean()
            test_acc = metric.result()
            test_ap = average_precision_score(np.hstack(test_labels_list), np.hstack(test_scores_list))


            print(f'Epoch {epoch:02d}')
            print(f"        Train: Loss: {train_loss:.4f}")
            if epoch >= 0:
                print(f'        Val:   Loss: {val_loss:.4f}    Acc: {val_acc:.4f}    AP:{val_ap:.4f} ')
                print(f'        Test:   Loss: {test_loss:.4f}    Acc: {test_acc:.4f}    AP:{test_ap:.4f}')


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='../../icdm_train')
    parser.add_argument('--labeled-class', type=str, default='item')
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Mini-batch size. If -1, use full graph training.")
    parser.add_argument("--fanout", type=int, default=30,
                        help="Fan-out of neighbor sampling.")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--h-dim", type=int, default=64,
                        help="number of hidden units")
    parser.add_argument("--in-dim", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--n-bases", type=int, default=8,
                        help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--early_stopping", type=int, default=10)
    parser.add_argument("--n-epoch", type=int, default=20)
    # test部分后续再增加
    parser.add_argument("--test-file", type=str, default="")

    parser.add_argument("--inference", type=bool, default=False)
    # parser.add_argument("--record-file", type=str, default="record.txt")
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--model-id", type=str, default="0")
    parser.add_argument("--device-id", type=int, default=0)

    args = parser.parse_args()

    main(args)
