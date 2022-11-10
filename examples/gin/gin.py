import argparse
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.loader import DataLoader
from gammagl.datasets import TUDataset
from gammagl.models import GINModel


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        train_logits = self.backbone_network(data.x, data.edge_index, data.batch)
        loss = self._loss_fn(train_logits, data.y)
        return loss


def calculate_acc(logits, y, metrics):
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


def main(args):
    print("loading dataset...")
    path = args.dataset_path + 'TU'
    dataset = TUDataset(path, name="MUTAG")

    dataset_unit = len(dataset) // 10
    train_dataset = dataset[2 * dataset_unit:]
    val_dataset = dataset[:dataset_unit]
    test_dataset = dataset[dataset_unit: 2 * dataset_unit]
    train_batch = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_batch = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_batch = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    net = GINModel(in_channels=dataset.num_features,
                   hidden_channels=args.hidden_dim,
                   out_channels=dataset.num_classes,
                   num_layers=args.num_layers,
                   name="GIN")

    # if tlx.BACKEND == film_utils.TORCH_BACKEND:
    #     net = net.to(device)

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)

    metrics = tlx.metrics.Accuracy()

    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)

    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        for data in train_batch:
            train_loss = train_one_step(data, data['y'])

        net.set_eval()

        for data in val_batch:
            val_logits = net(data.x, data.edge_index, data.batch)

            val_acc = calculate_acc(tlx.where(val_logits > 0, 1, 0), data.y, metrics)

            print("Epoch [{:0>3d}] ".format(epoch + 1) \
                  + "  train loss: {:.4f}".format(train_loss.item()) \
                  + "  val acc: {:.4f}".format(val_acc))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

    net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict')
    if tlx.BACKEND == 'torch':
        net.to(test_batch['x'].device)
    net.set_eval()
    for data in test_batch:
        test_logits = net(data.x, data.edge_index, data.batch)

        test_acc = calculate_acc(tlx.where(test_logits > 0, 1, 0), data.y, metrics)

        print("Test acc:  {:.4f}".format(test_acc))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=100, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=160, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.1, help="drop_rate")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='TUDataset', help='dataset(TUDataset)')
    parser.add_argument("--dataset_path", type=str, default=r'../../data/', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    parser.add_argument("--num_layers", type=int, default=5, help="num of gin layers")
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size of the data_loader")
    args = parser.parse_args()

    main(args)
