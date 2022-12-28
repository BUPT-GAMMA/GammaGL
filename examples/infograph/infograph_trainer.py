# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   infograph_trainer.py
@Time    :   2022/05/24 22:05:55
@Author  :   Yang Yuxiang
"""
import os

# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['CUDA_VISIBLE_DEVICES'] = ' '
# set your backend here, default `tensorflow`, you can choose 'paddle'、'tensorflow'、'torch'
from gammagl.datasets import TUDataset
from tqdm import tqdm
import tensorlayerx as tlx
import argparse
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.models.infograph import InfoGraph
from gammagl.loader import DataLoader
from infograph_eval import evaluate_embedding
# tlx.set_device("GPU", 4)


class Unsupervised_Loss(WithLoss):
    def __init__(self, net):
        super(Unsupervised_Loss, self).__init__(backbone=net, loss_fn=None)
        self.net = net

    def forward(self, data, label):
        loss = self._backbone(data.x, data.edge_index, data.batch)
        return loss


def main(args):
    # load  datasets
    dataset = TUDataset(args.dataset_path, args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    num_feature = max(dataset.num_features, 1)
    # build model
    net = InfoGraph(num_feature=num_feature, hid_feat=args.hid_dim, num_gc_layers=args.n_layers, prior=args.prior)
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    train_weights = net.trainable_weights
    loss_func = Unsupervised_Loss(net)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)
    best = 1e9
    acc = 0
    accuracies = {args.name_eval: []}
    log_interval = 1
    accuracies[args.name_eval]
    for epoch in tqdm(range(args.epochs)):
        loss_all = 0
        for data in dataloader:
            loss = train_one_step(data, tlx.convert_to_tensor([1]))
            loss_all += loss.item() * data.num_graphs
            net.set_train()
        print('===== Epoch {}, Loss {} ====='.format(epoch + 1, loss_all / len(dataloader)))
        if epoch % log_interval == 0:
            net.set_eval()
            x, y = net.get_embedding(dataloader)
            res = evaluate_embedding(x, y, args.name_eval)
            accuracies[args.name_eval].append(res)
            # save best model on evaluation set
            if loss < best:
                net.save_weights(args.best_model_path + "infograph.npz")
                best = loss
            if acc < res:
                acc = res
            print(accuracies)


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser(description='infograph')

    # data source params
    parser.add_argument('--dataset', type=str, default='MUTAG',
                        help='Name of dataset.eg:MUTAG,IMDB-BINARY,REDDIT-BINARY')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    # training params
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--log_interval', type=int, default=1, help='Interval between two evaluations.')

    # model params
    parser.add_argument('--n_layers', type=int, default=5,
                        help='Number of graph convolution layers before each pooling.')
    parser.add_argument('--hid_dim', type=int, default=32, help='Hidden layer dimensionalities.')
    parser.add_argument("--l2_coef", type=float, default=0., help="l2 loss coeficient")
    parser.add_argument('--prior', type=float, default=0.)
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")

    # evaluate embedding
    parser.add_argument('--name_eval', type=str, default='svc',
                        help='The name of classify to evaluate accuracy,supporting method:log,svc,linsvc,rf')
    args = parser.parse_args()

    main(args)
