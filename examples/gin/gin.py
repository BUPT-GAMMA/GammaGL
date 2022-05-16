"""
@File    :   gin.py
@Time    :   2022/03/07
@Author  :   Tianyu Zhao
"""

import os

os.environ['TL_BACKEND'] = 'tensorflow'  # set your backend here, default 'tensorflow'
# sys.path.insert(0, os.path.abspath('../')) # adds path2gammagl to execute in command line.

import argparse
import tensorflow as tf
import tensorlayerx as tlx
from gammagl.datasets import TUDataset
from gammagl.loader import DataLoader
from gammagl.models import GINModel


def train(model, train_loader, optimizer, train_weights):
    model.set_train()
    correct = 0
    all_loss = 0
    for data in train_loader:
        with tf.GradientTape() as tape:
            out = model(data.x, data.edge_index, data.batch)
            train_loss = tlx.losses.softmax_cross_entropy_with_logits(out, tlx.cast(data.y, dtype=tlx.int32))
            all_loss += train_loss
            pred = tlx.argmax(out, axis=1)
            correct += tlx.reduce_sum(tlx.cast((pred == data.y), dtype=tlx.int64))
        grad = tape.gradient(train_loss, train_weights)
        optimizer.apply_gradients(zip(grad, train_weights))
    print(all_loss)
    print(correct / len(train_loader.dataset))


def evaluate(model, loader):
    model.set_eval()  # disable dropout
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = tlx.argmax(out, axis=1)
        correct += tlx.reduce_sum(tlx.cast((pred == data.y), dtype=tlx.int64))
    return correct / len(loader.dataset)


def main(args):
    # load mutag dataset
    if str.lower(args.dataset) not in ['mutag']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    dataset = TUDataset(args.dataset_path, args.dataset, use_node_attr=True)

    train_dataset = dataset[:110]
    val_dataset = dataset[110:150]
    test_dataset = dataset[150:]
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

    # model = GINModel(name="GIN", input_dim=dataset.num_features, hidden=args.hidden_dim, out_dim=dataset.num_classes)
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
    # metrics = tlx.metrics.Accuracy()
    train_weights = model.trainable_weights
    '''
    data = {
        "x": x,
        "edge_index": edge_index,
        "train_mask": graph.train_mask,
        "test_mask": graph.test_mask,
        "num_nodes": graph.num_nodes,
    }
    '''

    best_val_acc = 0
    print('start training...')
    for epoch in range(args.n_epoch):
        train(model, train_loader, optimizer, train_weights)
        train_acc = evaluate(model=model, loader=train_loader)
        val_acc = evaluate(model=model, loader=val_loader)

        print("Epoch [{:0>3d}]  ".format(epoch + 1),
              "   train acc: {:.4f}".format(train_acc),
              "   val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(args.best_model_path + model.name + ".npz", format='npz_dict')

    model.load_weights(args.best_model_path + model.name + ".npz", format='npz_dict')
    test_acc = evaluate(model=model, loader=test_loader)
    print("Test acc:  {:.4f}".format(test_acc))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    # learning
    parser.add_argument("--lr", type=float, default=0.2, help="learning rate")
    parser.add_argument("--n_epoch", type=int, default=350, help="number of epoch")
    parser.add_argument("--final_dropout", type=float, default=0.01, help="final layer dropout")
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
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")

    # dataset
    parser.add_argument('--dataset', type=str, default='MUTAG', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")

    args = parser.parse_args()

    main(args)
