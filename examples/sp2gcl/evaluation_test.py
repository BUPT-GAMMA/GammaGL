import os
from tensorlayerx.model import TrainOneStep, WithLoss
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorlayerx as tlx
import tensorlayerx.nn as nn

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self.backbone_network(data['x'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(in_features=hid_dim, out_features=out_dim,W_init=tlx.initializers.xavier_uniform(), b_init=tlx.initializers.zeros())

    def forward(self, x):
        return self.linear(x)


def node_evaluation(emb, y, train_idx, test_idx, lr=1e-2, weight_decay=1e-4):
    nclass = y.max().item() + 1
    logreg = LogReg(hid_dim=emb.shape[1], out_dim=nclass)
    opt = tlx.optimizers.Adam(lr=lr, weight_decay=weight_decay)
    train_weights = logreg.trainable_weights
    loss = tlx.losses.softmax_cross_entropy_with_logits
    loss_func = SemiSpvzLoss(logreg, loss)
    train_one_step = TrainOneStep(loss_func, opt, train_weights)

    data = {
        'x': emb,
        'y': y,
        'train_idx':train_idx,
        'test_idx':test_idx
    }

    for epoch in range(500):
        logreg.set_train()
        loss = train_one_step(data=data, label=y)
        logreg.set_eval()
        test_logits = logreg(emb[test_idx])
        test_preds = tlx.argmax(test_logits, axis=1)
        test_acc = tlx.reduce_sum(test_preds == y[test_idx]).float() / test_idx.size(0)

    return test_acc
