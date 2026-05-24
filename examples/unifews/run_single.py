import os
import random
import argparse
import numpy as np

import tensorlayerx as tlx
from tensorlayerx import nn
from tensorlayerx.model import WithLoss, TrainOneStep

from gammagl.utils.logger_unifews import Logger, ModelLogger, prepare_opt
from gammagl.utils.loader_unifews import load_edgelist
import gammagl.utils.metric_unifews as metric
from gammagl.layers.conv.gcn_unifews import identity_n_norm
from gammagl.models.gnn_unifews import flops_modules_dict
import gammagl.models.mlp_unifews as mlp_model
import gammagl.models.gnn_unifews as gnn_model
import gammagl.models.gcn2_unifews as gcn2_model


class SemiSpvzLoss(WithLoss):
    """Semi-supervised loss wrapper that only computes loss on training nodes."""
    def __init__(self, net, loss_fn):
        super().__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        x, edge_idx, train_idx = data
        logits = self._backbone(x, edge_idx)
        train_logits = tlx.gather(logits, train_idx)
        return self._loss_fn(train_logits, label)



np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                    formatter=dict(float=lambda x: "%9.3e" % x))

# ========== Training settings
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--seed', type=int, default=11, help='Random seed.')
parser.add_argument('-v', '--dev', type=int, default=0, help='Device id.')
parser.add_argument('-c', '--config', type=str, default='cora', help='Config file name.')
parser.add_argument('-m', '--algo', type=str, default='gcn_unifews', help='Model name')
parser.add_argument('-n', '--suffix', type=str, default='', help='Save name suffix.')
parser.add_argument('-a', '--thr_a', type=float, default=0.5, help='Threshold of adj.')
parser.add_argument('-w', '--thr_w', type=float, default=0.5, help='Threshold of weight.')
parser.add_argument('-l', '--layer', type=int, default=2, help='Layer.')
parser.add_argument('--data', type=str, default='cora', help='dataset name')
parser.add_argument('--path', type=str, default='./data/', help='data path')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--patience', type=int, default=20, help='early stop patience')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--hidden', type=int, default=512, help='hidden dimension')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--inductive', action='store_true', default=False, help='inductive setting')
parser.add_argument('--multil', action='store_true', default=False, help='multi-label classification')

#args = prepare_opt(parser)
args = parser.parse_args()

if args.dev >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.dev)

# ========== random seed
random.seed(args.seed)
np.random.seed(args.seed)
tlx.set_seed(args.seed)

if not ('_' in args.algo):
    args.thr_a, args.thr_w = 0.0, 0.0


flag_run = f"{args.seed}-{args.thr_a:.1e}-{args.thr_w:.1e}"
logger = Logger(args.data, args.algo, flag_run=flag_run)
logger.save_opt(args)
model_logger = ModelLogger(logger,
                patience=args.patience,
                cmp='max',
                prefix='model'+args.suffix,
                storage='state')
stopwatch = metric.Stopwatch()

# ========== download data
adj, feat, labels, idx, nfeat, nclass = load_edgelist(
    datastr=args.data, datapath=args.path,
    inductive=args.inductive, multil=args.multil, seed=args.seed
)

if args.algo.split('_')[0] in ['gcn2']:
    model = gcn2_model.SandwitchThr(
        nlayer=args.layer, nfeat=nfeat, nhidden=args.hidden, nclass=nclass,
        thr_a=args.thr_a, thr_w=args.thr_w, dropout=args.dropout, layer=args.algo
    )
elif args.algo.split('_')[0] == 'mlp':
    model = mlp_model.MLP_unifews(
        nlayer=args.layer, nfeat=nfeat, nhidden=args.hidden, nclass=nclass,
        thr_w=args.thr_w, dropout=args.dropout, layer='mlp'
    )
else:
    model = gnn_model.GNNThr(
        nlayer=args.layer, nfeat=nfeat, nhidden=args.hidden, nclass=nclass,
        thr_a=args.thr_a, thr_w=args.thr_w, dropout=args.dropout, layer=args.algo
    )

model.reset_parameters()


adj['train'] = identity_n_norm(
    adj['train'], edge_weight=None, num_nodes=feat['train'].shape[0],
    rnorm=1, diag=None
)

if logger.lvl_config > 1:
    print(type(model).__name__, args.algo, args.thr_a, args.thr_w)
if logger.lvl_config > 2:
    print(model)

model_logger.register(model, save_init=False)

loss_fn = tlx.losses.sigmoid_cross_entropy if args.multil else tlx.losses.softmax_cross_entropy_with_logits
net_with_loss = SemiSpvzLoss(model, loss_fn)
optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
train_one_step = TrainOneStep(net_with_loss, optimizer, model.trainable_weights)

def train(x, edge_idx, y, idx_split, epoch, verbose=False):
    model.train()
    if epoch < args.epochs // 2:
        model.set_scheme('pruneall', 'pruneall')
    else:
        model.set_scheme('pruneall', 'pruneinc')

    stopwatch.reset()
    stopwatch.start()
    loss = train_one_step([x, edge_idx, idx_split], y)
    stopwatch.pause()
    return float(loss), stopwatch.time

def eval(x, edge_idx, y, idx_split, verbose=False):
    model.eval()
    model.set_scheme('keep', 'keep')
    calc = metric.F1Calculator(nclass)
    stopwatch.reset()

    stopwatch.start()
    output = model(x, edge_idx, node_lock=tlx.convert_to_tensor([]))[idx_split]
    stopwatch.pause()

    if args.multil:
        output = tlx.where(output > 0, 1.0, 0.0)
    else:
        output = tlx.argmax(output, axis=1)

    calc.update(y, output)
    res = calc.compute('micro')
    return res, stopwatch.time, output, y

def cal_flops(x, edge_idx, idx_split, verbose=False):
    return 0.0


time_tol, macs_tol = metric.Accumulator(), metric.Accumulator()
epoch_conv, acc_best = 0, 0

for epoch in range(1, args.epochs + 1):
    verbose = epoch % 1 == 0 and (logger.lvl_log > 0)
    loss_train, time_epoch = train(
        x=feat['train'], edge_idx=adj['train'],
        y=labels['train'], idx_split=idx['train'], epoch=epoch
    )
    time_tol.update(time_epoch)

    acc_val, _, _, _ = eval(
        x=feat['train'], edge_idx=adj['train'],
        y=labels['val'], idx_split=idx['val']
    )

    macs_epoch = cal_flops(feat['train'], adj['train'], idx['train'])
    macs_tol.update(macs_epoch)

    if verbose:
        res = f"Epoch:{epoch:04d} | loss:{loss_train:.4f}, val acc:{acc_val:.4f}, time:{time_tol.val:.4f}, macs:{macs_tol.val:.4f}"
        logger.print(res)

    acc_best = model_logger.save_best(acc_val, epoch=epoch)
    if not model_logger.is_early_stop(epoch=epoch):
        epoch_conv = max(0, epoch - model_logger.patience)


model = model_logger.load('best')
adj['test'] = identity_n_norm(
    adj['test'], edge_weight=None, num_nodes=feat['test'].shape[0],
    rnorm=1, diag=None
)

acc_test, time_test, outl, labl = eval(
    x=feat['test'], edge_idx=adj['test'],
    y=labels['test'], idx_split=idx['test']
)

macs_test = cal_flops(feat['test'], adj['test'], idx['test'])
numel_a, numel_w = model.get_numel()


print("="*60)
print(f"[Val] best acc: {acc_best:.5f}")
print(f"[Test] acc: {acc_test:.5f}")
print(f"[Test] MACs: {macs_test:.4f}G  |  Num adj: {numel_a:.3f}k  |  Num weight: {numel_w:.3f}k")
print("="*60)