import os
from time import time

# from tqdm import tqdm

os.environ['TL_BACKEND'] = 'paddle'  # set your backend here, default `tensorflow`
os.environ['CUDA_VISIBLE_DEVICES'] = ' '
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import argparse
import tensorlayerx as tlx
import numpy as np
from tensorlayerx.model import TrainOneStep, WithLoss
from preprocess import process_dataset
from gammagl.models.mvgrl import MVGRL, LogReg
from gammagl.datasets import Planetoid


class Unsupervised_Loss(WithLoss):
    def __init__(self, net):
        super(Unsupervised_Loss, self).__init__(backbone=net, loss_fn=None)

    def forward(self, data, label):
        loss = self._backbone(data["edge_index"], data["diff_adj"], data["feat"], data["shuf_feat"])
        # tensor.ref()是什么操作？？？
        return loss


def main(args):
    # load cora dataset
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    dataset.process()  # suggest to execute explicitly so far
    graph = dataset[0]

    diff_adj, feat, label, train_idx, val_idx, test_idx = process_dataset(args.dataset, graph)
    n_feat = feat.shape[1]
    # 建议加上处理的数据机上加一个 分类的个数
    # n_classes = graph.num_classes
    n_classes = 7

    lbl1 = tlx.ones((graph.num_nodes * 2,))
    lbl2 = tlx.zeros_like(lbl1)
    lbl = tlx.concat((lbl1, lbl2), axis=0)
    # create model
    net = MVGRL(feat.shape[1], args.hidden_dim)
    optim = tlx.optimizers.Adam(args.lr, weight_decay=args.l2_coef)

    train_weights = net.trainable_weights
    loss_func = Unsupervised_Loss(net)
    train_one_step = TrainOneStep(loss_func, optim, train_weights)
    # Step 4: Training epochs ================================================================ #
    best = float('inf')
    cnt_wait = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        shuf_idx = np.random.permutation(graph.num_nodes)
        feat = feat.numpy()
        shuf_feat = feat[shuf_idx, :]
        shuf_feat = shuf_feat
        data = {"edge_index": graph.edge_index, "diff_adj": tlx.convert_to_tensor(diff_adj),
                "feat": tlx.convert_to_tensor(feat),
                "shuf_feat": tlx.convert_to_tensor(shuf_feat)}

        loss = train_one_step(data, lbl)
        print('Epoch: {:02d}, Loss: {:0.4f}'.format(epoch, loss.item()))
        if loss < best:
            best = loss
            cnt_wait = 0
            net.save_weights(args.best_model_path + "MVGRL.npz")
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping')
            break
    net.load_weights(args.best_model_path + "MVGRL.npz")
    net.set_eval()

    embed = net.get_embedding(graph.edge_index, tlx.convert_to_tensor(diff_adj), tlx.convert_to_tensor(feat))

    train_embs = embed[graph.train_mask]
    test_embs = embed[graph.test_mask]
    y = tlx.convert_to_tensor(graph.y)
    train_lbls = y[graph.train_mask]
    train_lbls = tlx.argmax(train_lbls, axis=1)
    test_lbls = y[graph.test_mask]
    test_lbls = tlx.argmax(test_lbls, axis=1)
    accs = 0.
    for e in range(5):
        clf = LogReg(args.hidden_dim, y.shape[1])
        clf_opt = tlx.optimizers.Adam(lr=args.classifier_lr, weight_decay=args.clf_l2_coef)
        clf_loss_func = WithLoss(clf, tlx.losses.softmax_cross_entropy_with_logits)
        clf_train_one_step = TrainOneStep(clf_loss_func, clf_opt, clf.trainable_weights)
        # train classifier
        for a in range(300):
            clf.set_train()
            if os.environ['TL_BACKEND'] != 'tensorflow':
                clf_train_one_step(train_embs.detach(), train_lbls)
            else:
                clf_train_one_step(train_embs, train_lbls)
        test_logits = clf(test_embs)
        preds = tlx.argmax(test_logits, axis=1)

        acc = np.sum(preds.numpy() == test_lbls.numpy()) / test_lbls.shape[0]
        accs += acc
        # print("e", e, "acc", acc)
    print("avg_acc :{:.4f}".format(accs / 5))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--hidden_dim", type=int, default=512, help="dimention of hidden layers")
    parser.add_argument("--lr", type=float, default=0.001, help="learnin rate")
    parser.add_argument("--l2_coef", type=float, default=0., help="l2 loss coeficient")
    parser.add_argument("--n_epoch", type=int, default=1, help="number of epoch")
    parser.add_argument("--classifier_lr", type=float, default=1e-2, help="classifier learning rate")
    parser.add_argument("--clf_l2_coef", type=float, default=0.)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    main(args)
