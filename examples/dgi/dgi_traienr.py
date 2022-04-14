import os
os.environ['TL_BACKEND'] = 'tensorflow'
# set your backend here, default `tensorflow`, you can choose 'paddle'、'tensorflow'、'torch'

from tqdm import tqdm
import numpy as np
from gammagl.datasets import Planetoid
import tensorlayerx as tlx
import argparse
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.models.dgi import DGIModel
from gammagl.models.dgi import calc



class Unsupervised_Loss(WithLoss):
    def __init__(self, net):
        super(Unsupervised_Loss, self).__init__(backbone=net, loss_fn=None)

    def forward(self, data, label):
        loss = self._backbone(data["pos_feat"], data["neg_feat"], data["edge_index"], data["edge_weight"],
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
        self.fc = tlx.nn.Linear(out_features=num_classes, in_features=hid_feat)

    def forward(self, embed):
        return self.fc(embed)


def main(args):
    # load cora dataset
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    dataset.process()
    graph = dataset[0]

    # add self loop and calc Laplacian matrix
    row, col, edge_weight = calc(graph.edge_index, graph.num_nodes)
    edge_index = tlx.convert_to_tensor([row, col], dtype=tlx.int64)
    pos_feat = tlx.convert_to_tensor(graph.x)
    perm = tlx.convert_to_tensor(np.random.permutation(graph.num_nodes), dtype=tlx.int64)
    neg_feat = tlx.gather(pos_feat, perm)
    data = {
        "pos_feat": tlx.convert_to_tensor(pos_feat),
        "neg_feat": tlx.convert_to_tensor(neg_feat),
        "edge_index": tlx.convert_to_tensor(np.array([row, col], dtype=np.int64)),
        "edge_weight": tlx.convert_to_tensor(edge_weight),
        "num_node": graph.num_nodes
    }

    # build model
    net = DGIModel(in_feat=pos_feat.shape[1], hid_feat=args.hidden_dim,
                   act=tlx.nn.PRelu(args.hidden_dim, a_init=tlx.initializers.constant(0.25)))
    optimizer = tlx.optimizers.Adam(learning_rate=args.lr, weight_decay=args.l2_coef)
    train_weights = net.trainable_weights
    loss_func = Unsupervised_Loss(net)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    best = 1e9
    cnt_wait = 0
    for epoch in tqdm(range(args.n_epoch)):
        net.set_train()
        loss = train_one_step(data, None)
        print("loss :{:4f}".format(loss.item()))
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

    embed = net.gcn(tlx.convert_to_tensor(graph.x),
                    tlx.convert_to_tensor(edge_index),
                    tlx.convert_to_tensor(edge_weight),
                    graph.num_nodes)

    train_embs = embed[graph.train_mask]
    test_embs = embed[graph.test_mask]
    y = tlx.convert_to_tensor(graph.y)
    train_lbls = y[graph.train_mask]
    train_lbls = tlx.argmax(train_lbls, axis=1)
    test_lbls = y[graph.test_mask]
    test_lbls = tlx.argmax(test_lbls, axis=1)
    accs = 0.
    for e in range(args.classifier_epochs):
        # build clf model
        clf = Classifier(args.hidden_dim, 7)
        clf_opt = tlx.optimizers.Adam(learning_rate=args.classifier_lr, weight_decay=args.clf_l2_coef)
        clf_loss_func = WithLoss(clf, tlx.losses.softmax_cross_entropy_with_logits)
        clf_train_one_step = TrainOneStep(clf_loss_func, clf_opt, clf.trainable_weights)
        # train classifier
        for a in range(50):
            clf.set_train()
            if os.environ['TL_BACKEND'] != 'tensorflow':
                clf_train_one_step(train_embs.detach(), train_lbls)
            else :
                clf_train_one_step(train_embs, train_lbls)
        test_logits = clf(test_embs)
        preds = tlx.argmax(test_logits, axis=1)

        acc = np.sum(preds.numpy() == test_lbls.numpy()) / test_lbls.shape[0]
        accs += acc
        # print("e", e, "acc", acc)
    print("avg_acc :{:.4f}".format(accs / args.classifier_epochs))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=500, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=512, help="dimention of hidden layers")
    parser.add_argument("--classifier_lr", type=float, default=1e-2, help="classifier learning rate")
    parser.add_argument("--classifier_epochs", type=int, default=50)
    parser.add_argument("--l2_coef", type=float, default=0., help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='pubmed', help='dataset, pubmed, cora, citeseer')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--clf_l2_coef", type=float, default=0.)
    args = parser.parse_args()
    for i in range(2):
        main(args)
