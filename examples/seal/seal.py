import os
import os.path as osp
import time
import argparse
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import tensorlayerx as tlx
from sklearn.metrics import roc_auc_score
import numpy as np

from gammagl.datasets import Planetoid
from gammagl.loader import DataLoader
from gammagl.models import DGCNN, GINModel

from data import SEALDataset

class WithLoss(tlx.model.WithLoss):
    def __init__(self, net, loss_fn):
        super().__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, x, y):
        out = self._backbone(x.x, x.edge_index, x.batch)
        loss = self._loss_fn(tlx.squeeze(out, -1), y)
        return loss


def test(model, loader):
    y_pred, y_true = [], []
    for data in tqdm(loader):
        out = model(data.x, data.edge_index, data.batch)
        y_pred.append(tlx.convert_to_numpy(out).reshape(-1))
        y_true.append(tlx.convert_to_numpy(data.y))
    return roc_auc_score(np.concatenate(y_true), np.concatenate(y_pred))


def main(args):
    tlx.set_device('GPU', id=args.gpu_id)
    tlx.set_seed(args.seed)

    # data.
    path = osp.join(args.data_dir, 'Planetoid')
    dataset = Planetoid(path, name=args.dataset)
    data = dataset[0]
    train_dataset = SEALDataset(dataset, num_hops=args.hop, neg_sampling_ratio=args.neg_sampling_ratio, split='train', env=f'{tlx.BACKEND}_{args.dataset}')
    val_dataset = SEALDataset(dataset, num_hops=args.hop, neg_sampling_ratio=args.neg_sampling_ratio, split='val', env=f'{tlx.BACKEND}_{args.dataset}')
    test_dataset = SEALDataset(dataset, num_hops=args.hop, neg_sampling_ratio=args.neg_sampling_ratio, split='test', env=f'{tlx.BACKEND}_{args.dataset}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # model setting.
    if args.model == 'dgcnn':
        model = DGCNN(
            feature_dim=train_dataset.num_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            gcn_type=args.gcn_type,
            k=args.sort_k,
            train_dataset=train_dataset,
            dropout=args.dropout
        )
    else:
        model = GINModel(
            input_dim=train_dataset.num_features,
            hidden_dim=args.hidden_dim,
            output_dim=1,
            final_dropout=args.dropout,
            num_layers=args.num_layers,
            num_mlp_layers=args.num_mlp_layers,
            learn_eps=args.learn_eps,
            graph_pooling_type=args.graph_pooling_type,
            neighbor_pooling_type=args.neighbor_pooling_type,
            name="GIN"
        )
    optimizer = tlx.optimizers.Adam(lr=args.lr)
    criterion = tlx.losses.sigmoid_cross_entropy
    train_weights = model.trainable_weights

    net_with_loss = WithLoss(model, criterion)
    train_one_step = tlx.model.TrainOneStep(net_with_loss, optimizer, train_weights)


    # training.
    best_val_auc = best_auc = best_epoch = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        
        model.set_train()
        train_loss = 0
        for data in tqdm(train_loader):
            train_loss += train_one_step(data, data.y) * args.batch_size
        train_loss /= len(train_dataset)
        train_time = time.time()

        model.set_eval()
        if epoch % args.eval_steps == 0:
            val_auc = test(model, val_loader)
            test_auc = test(model, test_loader)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_auc = test_auc
                best_epoch = epoch
            eval_time = time.time()
            logging.info(f'epoch {epoch}, time took: train-{train_time - start_time:.1f}s, total-{eval_time - start_time:.1f}s')
            logging.info(f'train loss: {train_loss:.4f}, val auc:{val_auc:.4f}, test auc:{test_auc:.4f}')
            logging.info(f'!!! best_epoch: {best_epoch}, val auc:{best_val_auc:.4f}, test auc:{best_auc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SEAL")
    # data
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="Cora")

    # seal preprocess
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--neg_sampling_ratio", type=float, default=1.)

    # model and general setting"
    parser.add_argument("--model", type=str, default="dgcnn", choices=["dgcnn", "gin"])
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)

    # dgcnn
    parser.add_argument("--gcn_type", type=str, default="gcn")
    parser.add_argument("--sort_k", type=float, default=0.6)

    # gin
    parser.add_argument("--num_mlp_layers", type=int, default=2, help="number of MLP layers. 1 means linear model")
    parser.add_argument("--graph_pooling_type", type=str, default="sum", choices=["sum", "mean", "max"],
                        help="type of graph pooling: sum, mean or max")
    parser.add_argument("--neighbor_pooling_type", type=str, default="sum", choices=["sum", "mean", "max"],
                        help="type of neighboring pooling: sum, mean or max")
    parser.add_argument("--learn_eps", type=bool, default=False, help="learn the epsilon weighting")

    # training
    parser.add_argument("--epochs", type=int, default=81)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1313)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s %(levelname)s] %(message)s',
        handlers=[logging.FileHandler('1.log', mode='w'), logging.StreamHandler()]
    )

    main(args)