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


class SemiSpvzLoss(WithLoss):
    def __init__(self, net):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=None)

    def forward(self, data, label):
        loss = self._backbone(data["feature1"], data["feature2"], data["edge_index"], data["edge_weight"],
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
    row, col, weight = calc(graph.edge_index, graph.num_nodes)
    g = Graph(x=graph.x, edge_index=tlx.convert_to_tensor([row, col], dtype=tlx.int64), num_nodes=graph.num_nodes,
              y=graph.y)
    g.edge_weight = weight
    feature1 = graph.x
    perm = np.random.permutation(graph.num_nodes)
    feature2 = tlx.gather(feature1, perm)
    data = {
        "feature1": feature1,
        "feature2": feature2,
        "edge_index": tlx.stack([row, col]),
        "edge_weight": g.edge_weight,
        "num_node": g.num_nodes
    }

    # build model
    net = DGIModel(in_feat=feature1.shape[1], hid_feat=args.hidden_dim,
                   act=tlx.nn.PRelu(args.hidden_dim, a_init=tlx.initializers.constant(0.25)))
    optimizer = tlx.optimizers.Adam(args.lr)
    train_weights = net.trainable_weights
    loss_func = SemiSpvzLoss(net)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)
    best = 1e9
    cnt_wait = 0
    for epoch in tqdm(range(args.n_epoch)):
        net.set_train()
        # features1, features2, edge_index, edge_weight, num_nodes
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
    # model.load_weights(best_model_path+"DGI.npz")
    net.set_eval()
    # features, edge_index, edge_weight, num_nodes
    embed = net.gcn(g.x, g.edge_index, g.edge_weight, g.num_nodes)
    a = np.arange(0, g.num_nodes)
    train_embs = tlx.gather(embed, a[graph.train_mask])
    test_embs = tlx.gather(embed, a[graph.test_mask])
    y = g.y
    train_lbls = tlx.gather(y, a[graph.train_mask])
    test_lbls = tlx.gather(y, a[graph.test_mask])
    test_lbls = tlx.argmax(test_lbls, axis=1)
    accs = 0.
    for _ in range(args.classifier_epochs):
        clf = Classifier(args.hidden_dim, 7)
        clf_opt = tf.optimizers.Adam(learning_rate=args.classifier_lr)
        # best_acc = tf.zeros(1)
        for a in range(50):
            clf.set_train()
            with tf.GradientTape() as tape:
                train_logits = clf(train_embs)
                # loss = tlx.losses.softmax_cross_entropy_with_logits(train_logits, train_lbls)
                loss = tlx.reduce_mean(tlx.losses.softmax_cross_entropy_with_logits(output=train_logits,
                                                                                    target=tlx.argmax(train_lbls,
                                                                                                      axis=1)))
                grads = tape.gradient(loss, clf.trainable_weights)
                clf_opt.apply_gradients(zip(grads, clf.trainable_weights))
        test_logits = clf(test_embs)
        preds = np.argmax(test_logits, axis=1)

        acc = np.sum(preds == test_lbls) / test_lbls.shape[0]
        accs += acc
        print("e", _, "acc", acc)
    print("avg_acc :{:.4f}".format(accs / args.classifier_epochs))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=300, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=512, help="dimention of hidden layers")
    parser.add_argument("--classifier_lr", type=float, default=1e-2, help="classifier learning rate")
    parser.add_argument("--classifier_epochs", type=int, default=50)
    parser.add_argument("--l2_coef", type=float, default=0., help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='citeseer', help='dataset, pubmed, cora, citeseer')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--clf_l2_coef", type=float, default=0.)
    args = parser.parse_args()

    main(args)
