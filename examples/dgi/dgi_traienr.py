from tqdm import tqdm
import tensorflow as tf
import numpy as np
from gammagl.data import Graph
from gammagl.datasets import Planetoid
import tensorlayerx as tlx
import argparse

from tensorlayerx.model import TrainOneStep, WithLoss

from gammagl.models.dgi import DGIModel
from gammagl.models.grace import calc


class Unsupervised_Loss(WithLoss):
    def __init__(self, net):
        super(Unsupervised_Loss, self).__init__(backbone=net, loss_fn=None)

    def forward(self, data, label):
        loss = self._backbone(data["pos_feat"], data["neg_feat"], data["edge_index"], data["edge_weight"],
                              data["num_node"])
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in
                            self._backbone.trainable_weights]) * args.l2_coef  # only support for tensorflow backend
        return loss + l2_loss


class Classifier(tlx.nn.Module):
    def __init__(self, hid_feat, num_classes):
        super(Classifier, self).__init__()
        self.fc = tlx.layers.Dense(n_units=num_classes, in_channels=hid_feat)

    def forward(self, embed):
        return self.fc(embed)


def main(args):
    # load cora dataset
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]

    # add self loop and calc Laplacian matrix
    row, col, edge_weight = calc(graph.edge_index, graph.num_nodes)
    edge_index = tlx.convert_to_tensor([row, col], dtype=tlx.int64)
    pos_feat = graph.x
    perm = np.random.permutation(graph.num_nodes)
    neg_feat = tlx.gather(pos_feat, perm)
    data = {
        "pos_feat": pos_feat,
        "neg_feat": neg_feat,
        "edge_index": tlx.stack([row, col]),
        "edge_weight": edge_weight,
        "num_node": graph.num_nodes
    }

    # build model
    net = DGIModel(in_feat=pos_feat.shape[1], hid_feat=args.hidden_dim,
                   act=tlx.nn.PRelu(args.hidden_dim, a_init=tlx.initializers.constant(0.25)))
    optimizer = tlx.optimizers.Adam(args.lr)
    train_weights = net.trainable_weights
    loss_func = Unsupervised_Loss(net)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    best = 1e9
    cnt_wait = 0
    for epoch in tqdm(range(args.n_epoch)):
        net.set_train()
        loss = train_one_step(data, None)
        print("loss :{:4f}".format(loss))
        if loss < best:
            best = loss
            cnt_wait = 0
            net.save_weights(args.best_model_path + "DGI.npz")
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
    net.load_weights(args.best_model_path + "DGI.npz")
    net.set_eval()

    embed = net.gcn(graph.x, edge_index, edge_weight, graph.num_nodes)

    train_embs = embed[graph.train_mask]
    test_embs = embed[graph.test_mask]
    y = graph.y
    train_lbls = y[graph.train_mask]
    train_lbls = tlx.argmax(train_lbls, axis=1)
    test_lbls = y[graph.test_mask]
    test_lbls = tlx.argmax(test_lbls, axis=1)
    accs = 0.
    for e in range(args.classifier_epochs):
        # build clf model
        clf = Classifier(args.hidden_dim, 7)
        clf_opt = tf.optimizers.Adam(learning_rate=args.classifier_lr)
        clf_loss_func = WithLoss(clf, tlx.losses.softmax_cross_entropy_with_logits)
        clf_train_one_step = TrainOneStep(clf_loss_func, clf_opt, clf.trainable_weights)
        for a in range(50):
            clf.set_train()
            clf_train_one_step(train_embs, train_lbls)
        test_logits = clf(test_embs)
        preds = np.argmax(test_logits, axis=1)
        acc = np.sum(preds == test_lbls) / test_lbls.shape[0]
        accs += acc
        print("e", e, "acc", acc)
    print("avg_acc :{:.4f}".format(accs / args.classifier_epochs))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=512, help="dimention of hidden layers")
    parser.add_argument("--classifier_lr", type=float, default=1e-2, help="classifier learning rate")
    parser.add_argument("--classifier_epochs", type=int, default=50)
    parser.add_argument("--l2_coef", type=float, default=0., help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset, pubmed, cora, citeseer')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--clf_l2_coef", type=float, default=0.)
    args = parser.parse_args()

    main(args)
