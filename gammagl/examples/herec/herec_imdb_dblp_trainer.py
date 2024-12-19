import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR

import os.path as osp
import tensorlayerx as tlx
from gammagl.datasets import IMDB, DBLP
from gammagl.models import HERec
from tensorlayerx.model import WithLoss, TrainOneStep
from sklearn.linear_model import LogisticRegression
from gammagl.utils import mask_to_index

class Unsupervised_Loss(WithLoss):
    def __init__(self, net, loss_fn):
        super(Unsupervised_Loss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self.backbone_network(data["pos_rw"], data["neg_rw"])
        loss = self._loss_fn(logits, label)

        return loss


def calculate_acc(train_z, train_y, test_z, test_y, solver='lbfgs', multi_class='auto', max_iter=150):
    train_z = tlx.convert_to_numpy(train_z)
    train_y = tlx.convert_to_numpy(train_y)
    test_z = tlx.convert_to_numpy(test_z)
    test_y = tlx.convert_to_numpy(test_y)

    clf = LogisticRegression(solver=solver, multi_class=multi_class, max_iter=max_iter).fit(train_z, train_y)
    return clf.score(test_z, test_y)


def main(args, log_steps=10):
    # load datasets
    if str.lower(args.dataset) not in ['imdb', 'dblp']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    targetType = {
        'imdb': 'movie',
        'dblp': 'author'
    }
    if str.lower(args.dataset) == 'imdb':
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/IMDB')
        dataset = IMDB(path)
        graph = dataset[0]

        metapath = [
            ('actor', 'to', 'movie'),
            ('movie', 'to', 'director'),
            ('director', 'to', 'movie'),
            ('movie', 'to', 'actor')
        ]
    elif str.lower(args.dataset) == 'dblp':
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
        dataset = DBLP(path)
        graph = dataset[0]

        metapath = [
            ('author', 'to', 'paper'),
            ('paper', 'to', 'term'),
            ('term', 'to', 'paper'),
            ('paper', 'to', 'author')
        ]

    graph[targetType[str.lower(args.dataset)]].train_mask = tlx.convert_to_tensor(
        graph[targetType[str.lower(args.dataset)]].train_mask)
    graph[targetType[str.lower(args.dataset)]].val_mask = tlx.convert_to_tensor(
        graph[targetType[str.lower(args.dataset)]].val_mask)
    graph[targetType[str.lower(args.dataset)]].test_mask = tlx.convert_to_tensor(
        graph[targetType[str.lower(args.dataset)]].test_mask)

    train_idx = mask_to_index(graph[targetType[str.lower(args.dataset)]].train_mask)
    val_idx = mask_to_index(graph[targetType[str.lower(args.dataset)]].val_mask)
    test_idx = mask_to_index(graph[targetType[str.lower(args.dataset)]].test_mask)

    model = HERec(graph.edge_index_dict,
                  embedding_dim=args.embedding_dim,
                  metapath=metapath,
                  walk_length=args.walk_length,
                  context_size=args.window_size,
                  walks_per_node=args.num_walks,
                  num_negative_samples=args.num_negative_samples,
                  target_type=targetType[args.dataset],
                  dataset=args.dataset,
                  name="HERec")
    loader = model.loader(batch_size=args.batch_size, shuffle=True)

    optimizer = tlx.optimizers.Adam(lr=args.lr)
    train_weights = model.trainable_weights
    loss_func = Unsupervised_Loss(net=model, loss_fn=tlx.losses.absolute_difference_error)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "y": tlx.convert_to_tensor(graph[targetType[args.dataset]].y),
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(loader):
            data['pos_rw'] = pos_rw
            data['neg_rw'] = neg_rw
            model.set_train()
            train_loss = train_one_step(data, tlx.convert_to_tensor(0, dtype=tlx.float32))
            total_loss += train_loss.item()
            if (i + 1) % log_steps == 0:
                model.set_eval()
                z = model.campute()
                val_acc = calculate_acc(tlx.gather(z, data['train_idx']), tlx.gather(data['y'], data['train_idx']),
                                        tlx.gather(z, data['val_idx']), tlx.gather(data['y'], data['val_idx']),
                                        max_iter=150)

                print((f'Epoch: {epoch}, Step: {i + 1:02d}/{len(loader)}, '
                       f'Loss: {total_loss / log_steps:.4f}, '
                       f'Acc: {val_acc:.4f}'))
                total_loss = 0

                # save best model on evaluation set
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model.save_weights(
                        args.best_model_path + tlx.BACKEND + "_" + args.dataset + "_" + model.name + ".npz",
                        format='npz_dict')

    model.load_weights(args.best_model_path + tlx.BACKEND + "_" + args.dataset + "_" + model.name + ".npz",
                       format='npz_dict', skip=True)
    model.set_eval()
    z = model.campute()
    test_acc = calculate_acc(tlx.gather(z, data['train_idx']), tlx.gather(data['y'], data['train_idx']),
                             tlx.gather(z, data['test_idx']), tlx.gather(data['y'], data['test_idx']),
                             max_iter=150)

    print("Test acc:  {:.4f}".format(test_acc))
    return test_acc


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imdb', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--n_epoch", type=int, default=50, help="number of epoch")
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--walk_length", type=int, default=50)
    parser.add_argument("--num_walks", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=7)
    parser.add_argument("--train_ratio", type=float, default=0.1)
    parser.add_argument("--num_negative_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")
    main(args, log_steps=10)
