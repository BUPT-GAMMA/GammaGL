import sys
import os

sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.datasets.ppi import PPI
from gammagl.models import FILMModel
from gammagl.loader import DataLoader
from sklearn.metrics import f1_score


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        train_logits = self.backbone_network(data['x'], data['edge_index'])
        loss = self._loss_fn(train_logits, tlx.cast(data['y'], dtype=tlx.float32))
        return loss


def calculate_acc(logits, y, metrics):
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


def main(args):
    print("loading ppi dataset...")
    train_dataset = PPI(args.dataset_path)
    val_dataset = PPI(args.dataset_path, split='val')
    test_dataset = PPI(args.dataset_path, split='test')

    batch_size = int(args.batch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    net = FILMModel(in_channels=train_dataset.num_node_features,
                    hidden_dim=args.hidden_dim,
                    out_channels=train_dataset.num_classes,
                    num_layers=args.num_layers,
                    drop_rate=args.drop_rate,
                    name="FILM")

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)

    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.sigmoid_cross_entropy)

    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        for batch in train_loader:
            train_loss = train_one_step(batch, batch['y'])

        net.set_eval()

        for batch in val_loader:
            val_logits = net(batch['x'], batch['edge_index'])

            val_y = batch['y']
            pred = tlx.where(val_logits > 0, 1, 0)

            val_f1 = f1_score(val_y.cpu(), pred.cpu(), average='micro')

            print("Epoch [{:0>3d}] ".format(epoch + 1) \
                  + "  train loss: {:.4f}".format(train_loss.item()) \
                  + "  val f1-micro: {:.4f}".format(val_f1))

            if val_f1 > best_val_acc:
                best_val_acc = val_f1
                net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

    net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict')
    net.set_eval()
    for batch in test_loader:
        test_logits = net(batch['x'], batch['edge_index'])

        test_y = batch['y']
        pred = tlx.where(test_logits > 0, 1, 0)

        test_f1 = f1_score(test_y.cpu(), pred.cpu(), average='micro')
        print("Test f1-micro:  {:.4f}".format(test_f1))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=2000, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=320, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.1, help="drop_rate")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='ppi', help='dataset(ppi)')
    parser.add_argument("--dataset_path", type=str, default=r'../../../data', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    parser.add_argument("--batch_size", type=int, default=2, help="batch_size of dataloader")
    parser.add_argument("--num_layers", type=int, default=4, help="num of film layers")
    args = parser.parse_args()

    main(args)
