import os

from tensorlayerx.model import WithLoss, TrainOneStep

os.environ['TL_BACKEND'] = 'tensorflow'  # set your backend here, default 'tensorflow'
# sys.path.insert(0, os.path.abspath('../')) # adds path2gammagl to execute in command line.
import numpy as np
import argparse
import tensorflow as tf
import tensorlayerx as tlx
from gammagl.datasets import TUDataset
from gammagl.models import GINModel
from ginDataLoader import GINDataLoader


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self._backbone(data['x'], data['edge_index'], data['batch'])
        loss = self._loss_fn(logits, label)

        return loss



def train(model, trainloader, train_one_step):
    model.set_train()
    all_loss = 0
    for data in trainloader:
        rundata = {
            "x": data.x,
            "edge_index": data.edge_index,
            "batch": data.batch
        }
        train_loss = train_one_step(data=rundata, label=tlx.cast(data.y, dtype=tlx.int32))
        all_loss += train_loss

    print(len(trainloader.sampler))
    all_loss = all_loss / len(trainloader.sampler)
    print(all_loss)

    return all_loss


def evaluate(model, dataloader, criterion):
    model.set_eval()  # disable dropout
    total = 0
    total_loss = 0
    total_correct = 0
    for data in dataloader:
        out = model(data.x, data.edge_index, data.batch)
        total += len(data.y)
        pred = tlx.argmax(out, axis=1)
        total_correct += tlx.reduce_sum(tlx.cast((pred == data.y), dtype=tlx.int64))
        loss = criterion(out, tlx.cast(data.y, dtype=tlx.int32))
        total_loss += loss * len(data.y)

    e_loss, e_acc = total_loss / total, total_correct / total

    model.set_train()
    return e_loss, e_acc


def main(args):
    np.random.seed(seed=args.seed)

    # load mutag dataset
    if str.lower(args.dataset) not in ['mutag', 'proteins', 'collab']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    dataset = TUDataset(args.dataset_path, args.dataset, use_node_attr=True)

    trainloader, validloader = GINDataLoader(dataset=dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True,
                                             split_name='fold10', fold_idx=args.fold_idx).train_valid_loader()

    model = GINModel(input_dim=dataset.num_features,
                     hidden_dim=args.hidden_dim,
                     output_dim=dataset.num_classes,
                     final_dropout=args.final_dropout,
                     num_layers=args.num_layers,
                     num_mlp_layers=args.num_mlp_layers,
                     learn_eps=args.learn_eps,
                     graph_pooling_type=args.graph_pooling_type,
                     neighbor_pooling_type=args.neighbor_pooling_type,
                     name="GIN"
                     )
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    criterion = tlx.losses.softmax_cross_entropy_with_logits
    scheduler = tlx.optimizers.lr.StepDecay(learning_rate=args.lr, step_size=50, gamma=0.5)
    train_weights = model.trainable_weights

    loss_func = SemiSpvzLoss(net=model, loss_fn=criterion)
    train_one_step = TrainOneStep(net_with_loss=loss_func, optimizer=optimizer, train_weights=train_weights)

    # metrics = tlx.metrics.Accuracy()


    best_val_acc = 0
    min_val_loss = 1
    print('start training...')
    for epoch in range(args.n_epoch):
        train(model=model, trainloader=trainloader, train_one_step=train_one_step)
        scheduler.step()

        train_loss, train_acc = evaluate(model=model, dataloader=trainloader, criterion=criterion)
        valid_loss, valid_acc = evaluate(model=model, dataloader=validloader, criterion=criterion)

        print("Epoch [{:0>3d}]  ".format(epoch + 1),
              "   train acc: {:.4f}".format(train_acc),
              "   val acc: {:.4f}".format(valid_acc))

        if not args.best_model_path == "":
            with open(args.best_model_path + model.name + dataset.name + '.txt', 'a') as f:
                f.write('%s %s %s %f %f %f %f %s' % (
                    args.learn_eps,
                    args.neighbor_pooling_type,
                    args.graph_pooling_type,
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc,
                    epoch,
                ))
                f.write("\n")

        # save best model on evaluation set
        if valid_acc >= best_val_acc and valid_loss < min_val_loss:
            best_val_acc = valid_acc
            min_val_loss = valid_loss
            model.save_weights(args.best_model_path + model.name + ".npz", format='npz_dict')

    # model.load_weights(args.best_model_path + model.name + ".npz", format='npz_dict')

    print("Test acc:  {:.4f}".format(best_val_acc))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    # learning
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--n_epoch", type=int, default=35, help="number of epoch")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument("--final_dropout", type=float, default=0.5, help="final layer dropout")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")

    # graph
    parser.add_argument("--graph_pooling_type", type=str, default="sum", choices=["sum", "mean", "max"],
                        help="type of graph pooling: sum, mean or max")
    parser.add_argument("--neighbor_pooling_type", type=str, default="sum", choices=["sum", "mean", "max"],
                        help="type of neighboring pooling: sum, mean or max")
    parser.add_argument("--learn_eps", type=bool, default=False, help="learn the epsilon weighting")

    # net
    parser.add_argument("--num_layers", type=int, default=5, help="numbers of layers")
    parser.add_argument("--num_mlp_layers", type=int, default=2, help="number of MLP layers. 1 means linear model")
    parser.add_argument("--hidden_dim", type=int, default=64, help="dimension of hidden layers")

    # dataset
    parser.add_argument('--dataset', type=str, default='MUTAG', help='dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training and validation')
    parser.add_argument('--fold_idx', type=int, default=0, help='the index(<10) of fold in 10-fold validation.')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")

    args = parser.parse_args()

    main(args)
