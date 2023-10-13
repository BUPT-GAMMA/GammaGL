import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
import argparse
import tensorlayerx as tlx
from gammagl.utils import mask_to_index
from tensorlayerx.model import TrainOneStep, WithLoss
import gammagl.transforms as T
from gammagl.datasets import HGBDataset, IMDB
from gammagl.models import ieHGCNModel

# This model only support dataset DBLP and IMDB.
targetType = {
    'imdb': 'movie',
    'dblp_hgb': 'author'
}

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x_dict'], data['edge_index_dict'], data['num_nodes_dict'])
        train_logits = tlx.gather(logits[targetType[str.lower(args.dataset)]], data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss


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
    if (str.lower(args.dataset) not in ['imdb', 'dblp_hgb']):
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    # load dataset
    if str.lower(args.dataset) == 'imdb':
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../IMDB')
        metapaths = [[('movie', 'actor'), ('actor', 'movie')],
                     [('movie', 'director'), ('director', 'movie')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edges=True,
                                   drop_unconnected_nodes=True)
        dataset = IMDB(path, transform=transform)
        graph = dataset[0]
        y = graph[targetType[str.lower(args.dataset)]].y
        num_classes = max(y) + 1
        # for mindspore, it should be passed into node indices
        train_idx = mask_to_index(graph[targetType[str.lower(args.dataset)]].train_mask)
        test_idx = mask_to_index(graph[targetType[str.lower(args.dataset)]].test_mask)
        val_idx = mask_to_index(graph[targetType[str.lower(args.dataset)]].val_mask)
        num_nodes_dict = {targetType[str.lower(args.dataset)]: graph[targetType[str.lower(args.dataset)]].num_nodes}

    else:
        if tlx.BACKEND == 'tensorflow':
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../DBLP')
            metapaths = [[('author', 'paper'), ('paper', 'author')],
                         [('author', 'paper'), ('paper', 'term'), ('term', 'paper'), ('paper', 'author')],
                         [('author', 'paper'), ('paper', 'venue'), ('venue', 'paper'), ('paper', 'author')]]
            transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edges=True,
                                       drop_unconnected_nodes=True)

            dataset = HGBDataset(path, args.dataset, transform=transform)
            graph = dataset[0]
            y = graph[targetType[str.lower(args.dataset)]].y
            num_classes = (max(y) - min(y)) + 1
            val_ratio = 0.2
            train = mask_to_index(graph[targetType[str.lower(args.dataset)]].train_mask)
            split = int(train.shape[0] * val_ratio)
            train_idx = train[split:]
            val_idx = train[:split]
            test_idx = mask_to_index(graph[targetType[str.lower(args.dataset)]].test_mask)
            num_nodes_dict = {targetType[str.lower(args.dataset)]: graph[targetType[str.lower(args.dataset)]].num_nodes}
        else:
            dataset = HGBDataset(args.dataset_path, args.dataset)
            graph = dataset[0]
            y = graph[targetType[str.lower(args.dataset)]].y
            num_classes = (max(y) - min(y)) + 1
            val_ratio = 0.2
            train = mask_to_index(graph[targetType[str.lower(args.dataset)]].train_mask)
            split = int(train.shape[0] * val_ratio)
            train_idx = train[split:]
            val_idx = train[:split]
            test_idx = mask_to_index(graph[targetType[str.lower(args.dataset)]].test_mask)
            num_nodes_dict = {'author': graph['author'].num_nodes, 'paper': graph['paper'].num_nodes,
                              'term': graph['term'].num_nodes, 'venue': graph['venue'].num_nodes}

    if tlx.BACKEND == 'tensorflow':
        edge_index_dict = graph.edge_index_dict
    else:
        edge_index_dict = {graph.edge_types[i]: graph.edge_stores[i]['edge_index'] for i in
                           range(len(graph.edge_stores))}

    # for IMDB: train test val = 400, 3478, 400
    # for DBLP: train test val = 974, 1420, 243
    data = {
        "x_dict": graph.x_dict,
        "y": y,
        "edge_index_dict": edge_index_dict,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes_dict": num_nodes_dict,
    }

    net = ieHGCNModel(
        num_layers=args.num_layers,
        in_channels={node_type: node_shape.shape[1] for node_type, node_shape in graph.x_dict.items()},
        hidden_channels=args.hidden_channels,
        out_channels=num_classes,
        attn_channels=args.attn_channels,
        metadata=graph.metadata(),
        batchnorm=False,
        add_bias=True,
        dropout_rate=args.dropout_rate,
        name='iehgcn',
    )

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = tlx.losses.softmax_cross_entropy_with_logits
    semi_spvz_loss = SemiSpvzLoss(net, loss_func)
    train_one_step = TrainOneStep(semi_spvz_loss, optimizer, train_weights)

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, y)
        net.set_eval()
        logits = net(data['x_dict'], data['edge_index_dict'], data['num_nodes_dict'])
        val_logits = tlx.gather(logits[targetType[str.lower(args.dataset)]], data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        val_acc = calculate_acc(val_logits, val_y, metrics)

        print("Epoch [{:0>3d}]  ".format(epoch + 1)
              + "   train_loss: {:.4f}".format(train_loss.item())
              + "   val_acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

    net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

    net.set_eval()
    logits = net(data['x_dict'], data['edge_index_dict'], data['num_nodes_dict'])
    test_logits = tlx.gather(logits[targetType[str.lower(args.dataset)]], data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])

    if tlx.BACKEND == 'torch':
        test_logits = test_logits.detach().numpy()  # torch
    else:
        test_logits = test_logits.numpy()

    import numpy as np
    test_logits = np.argmax(test_logits, axis=1)

    if tlx.BACKEND == 'torch':
        test_y = test_y.detach().numpy()  # torch
    else:
        test_y = test_y.numpy()

    from sklearn.metrics import f1_score
    macro_f1 = f1_score(y_true=test_y, y_pred=test_logits, average='macro')
    micro_f1 = f1_score(y_true=test_y, y_pred=test_logits, average='micro')
    print("Macro-F1:  {:.4f}".format(macro_f1))
    print("Micro-F1:  {:.4f}".format(micro_f1))
    return macro_f1


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset, not work")
    parser.add_argument('--dataset', type=str, default='DBLP_hgb', help='dataset, IMDB or DBLP')
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--n_epoch", type=int, default=30, help="number of epoch")
    parser.add_argument("--num_layers", type=int, default=4, help="number of layers")
    parser.add_argument("--hidden_channels", type=int, default=[64, 32, 16], help="dimention of hidden layers")
    parser.add_argument("--attn_channels", type=int, default=32, help="dimention of attention layers")
    parser.add_argument("--l2_coef", type=float, default=0.0005, help="l2 loss coeficient")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="dropout_rate")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    args = parser.parse_args()

    main(args)
