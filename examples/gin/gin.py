
import numpy
import os
import sys

sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.

import argparse
import numpy as np

import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.loader import DataLoader
from gammagl.datasets import TUDataset
from gammagl.models import GINModel

tlx.set_device('GPU', 5)


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        train_logits = self.backbone_network(data.x, data.edge_index, data.batch)
        # train_logits = self.backbone_network(data.x, data.edge_index, None, data.batch.shape[0], data.batch)
        loss = self._loss_fn(train_logits, data.y)
        return loss



def main(args):
    print("loading dataset...")

    path = args.dataset_path
    dataset = TUDataset(path, name=args.dataset)


    dataset_unit = len(dataset) // 10
    train_dataset = dataset[2 * dataset_unit:]
    val_dataset = dataset[:dataset_unit]
    test_dataset = dataset[dataset_unit: 2 * dataset_unit]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    net = GINModel(in_channels=max(dataset.num_features, 1),

                   hidden_channels=args.hidden_dim,
                   out_channels=dataset.num_classes,
                   num_layers=args.num_layers,
                   name="GIN")


    # net = GCNModel(feature_dim=max(dataset.num_features, 1),
    #                hidden_dim=args.hidden_dim,
    #                num_class=dataset.num_classes,
    #                drop_rate=args.drop_rate,
    #                num_layers=args.num_layers,
    #                name="GCN")

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)


    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)

    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()

        for data in train_loader:
            train_loss = train_one_step(data, data.y)

        net.set_eval()

        total_correct = 0
        for data in val_loader:
            val_logits = net(data.x, data.edge_index, data.batch)
            # val_logits = net(data.x, data.edge_index, None, data.batch.shape[0], data.batch)
            pred = tlx.argmax(val_logits, axis=-1)
            total_correct += int((numpy.sum(tlx.convert_to_numpy(pred == data.y).astype(int))))

        val_acc = total_correct / len(val_dataset)

        print("Epoch [{:0>3d}] ".format(epoch + 1) \
              + "  train loss: {:.4f}".format(train_loss.item()) \
              + "  val acc: {:.4f}".format(val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

    net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict')
    net.set_eval()
    total_correct = 0
    for data in test_loader:
        test_logits = net(data.x, data.edge_index, data.batch)
        # test_logits = net(data.x, data.edge_index, None, data.batch.shape[0], data.batch)
        pred = tlx.argmax(test_logits, axis=-1)
        total_correct += int((numpy.sum(tlx.convert_to_numpy(pred == data['y']).astype(int))))
    test_acc = total_correct / len(test_dataset)

    print("Test acc:  {:.4f}".format(test_acc))



if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--n_epoch", type=int, default=100, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=32, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.1, help="drop_rate")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    # parser.add_argument('--dataset', type=str, default='MUTAG', help='dataset(MUTAG/IMDB-BINARY/REDDIT-BINARY)')
    # parser.add_argument('--dataset', type=str, default='IMDB-BINARY', help='dataset(MUTAG/IMDB-BINARY/REDDIT-BINARY)')
    # parser.add_argument('--dataset', type=str, default='REDDIT-BINARY', help='dataset(MUTAG/IMDB-BINARY/REDDIT-BINARY)')
    parser.add_argument("--dataset_path", type=str, default=r'../../../data', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    parser.add_argument("--num_layers", type=int, default=5, help="num of gin layers")
    parser.add_argument("--batch_size", type=int, default=100, help="batch_size of the data_loader")

    args = parser.parse_args()

    main(args)
