import math
import os

# os.environ['TL_BACKEND'] = 'torch'
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from gammagl.utils import add_self_loops, calc_gcn_norm, mask_to_index, remove_self_loops

import argparse
from tqdm import tqdm

from gammagl.datasets import Planetoid
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.models.dgi import DGIModel


class Unsupervised_Loss(WithLoss):
    def __init__(self, net):
        super(Unsupervised_Loss, self).__init__(backbone=net, loss_fn=None)

    def forward(self, data, label):
        loss = self._backbone(data["x"], data["edge_index"], data["edge_weight"],
                              data["num_node"])
        return loss


class Clf_Loss(WithLoss):
    def __init__(self, net, lossfn):
        super(Clf_Loss, self).__init__(backbone=net, loss_fn=lossfn)

    def forward(self, data, label):
        loss = self._backbone(data)
        return loss


class Classifier(tlx.nn.Module):
    def __init__(self, hid_feat, num_classes):
        super(Classifier, self).__init__()
        # import math
        init = tlx.nn.initializers.HeNormal(a=math.sqrt(5))
        self.fc = tlx.nn.Linear(out_features=num_classes, in_features=hid_feat, W_init=init)

    def forward(self, embed):
        return self.fc(embed)


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
    # load datasets
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]

    # for mindspore, it should be passed into node indices
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)

    # add self loop and calc Laplacian matrix
    edge_index = graph.edge_index
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes, n_loops=args.self_loops)
    edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, graph.num_nodes))

    data = {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "num_node": graph.num_nodes,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "x": graph.x,
        "y": graph.y
    }

    # build model
    net = DGIModel(in_feat=dataset.num_node_features, hid_feat=args.hidden_dim,
                   act=tlx.nn.PRelu(args.hidden_dim))
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    train_weights = net.trainable_weights
    loss_func = Unsupervised_Loss(net)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    best = 1e9
    cnt_wait = 0
    for _ in tqdm(range(args.n_epoch)):
        net.set_train()
        # label is None in unsupervised learning. graph.y will not be used
        loss = train_one_step(data=data, label=graph.y)
        print("loss :{:4f}".format(loss.item()))
        if loss < best:
            best = loss
            cnt_wait = 0
            net.save_weights(args.best_model_path + "DGI_" + args.dataset + ".npz")
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
    net.load_weights(args.best_model_path + "DGI_" + args.dataset + ".npz")
    net.set_eval()
    if tlx.BACKEND == 'torch':
        net.to(data['x'].device)
    embed = net.gcn(data['x'],
                    data['edge_index'],
                    data['edge_weight'],
                    graph.num_nodes)

    train_embs = tlx.gather(embed, data['train_idx'])
    test_embs = tlx.gather(embed, data['test_idx'])

    if tlx.BACKEND != 'tensorflow':
        # todo: support ms
        train_embs = train_embs.detach()

    train_lbls = tlx.gather(data['y'], data['train_idx'])
    test_lbls = tlx.gather(data['y'], data['test_idx'])
    accs = 0.
    for e in range(args.num_evaluation):
        # build clf model
        clf = Classifier(args.hidden_dim, dataset.num_classes)
        clf_opt = tlx.optimizers.Adam(lr=args.classifier_lr, weight_decay=args.clf_l2_coef)
        clf_loss_func = WithLoss(clf, tlx.losses.softmax_cross_entropy_with_logits)
        clf_train_one_step = TrainOneStep(clf_loss_func, clf_opt, clf.trainable_weights)
        # train classifier
        for a in range(args.classifier_epochs):
            clf.set_train()
            clf_train_one_step(train_embs, train_lbls)
        test_logits = clf(test_embs)
        acc = calculate_acc(test_logits, test_lbls, tlx.metrics.Accuracy())
        print(acc)
        accs += acc
    print("avg_acc :{:.4f}".format(accs / args.num_evaluation))
    return accs / args.num_evaluation


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.002, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=1000, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=512, help="dimention of hidden layers")
    parser.add_argument("--classifier_lr", type=float, default=1e-2, help="classifier learning rate")
    parser.add_argument("--classifier_epochs", type=int, default=100, help="the epoch to train classifier")
    parser.add_argument("--l2_coef", type=float, default=0., help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset, pubmed, cora, citeseer')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--clf_l2_coef", type=float, default=0.)
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    parser.add_argument("--num_evaluation", type=int, default=50, help="number of evaluate classifier")
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    # main(args)
    accs = []
    print(args)
    for i in range(5):
        accs.append(main(args))
    import numpy as np
    print("mean: {:.4f} \u00b1 {:4f}".format(np.mean(accs), np.std(accs)))

