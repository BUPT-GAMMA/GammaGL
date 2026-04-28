import os
os.environ['TL_BACKEND'] = 'torch'

import random
import argparse
import numpy as np

import tensorlayerx as tlx
from tensorlayerx import nn

from utils.logger_gamma import Logger, ModelLogger, prepare_opt
from utils.loader_gamma import load_edgelist
import utils.metric_gamma as metric
from archs_gamma import identity_n_norm, flops_modules_dict
import archs_gamma.models_gamma as models


class PureTLXAdam:
    def __init__(self, lr=0.001, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = []
        self.v = []
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-7

    def gradient(self, loss, weights):
        
        if hasattr(loss, 'backward'):
            loss.backward()
        
        grads = []
        for w in weights:
            grad = w.grad if hasattr(w, 'grad') else tlx.zeros_like(w)
            grads.append(grad)
        return grads

    def apply_gradients(self, grads_and_vars):
       
        grads_and_vars = list(grads_and_vars)
        if not grads_and_vars:
            return
        
        self.t += 1
        bias_correction1 = 1.0 - self.beta1**self.t
        bias_correction2 = 1.0 - self.beta2**self.t
        step_size = self.lr * (bias_correction2**0.5) / bias_correction1
        
        if len(self.m) != len(grads_and_vars):
            self.m = [tlx.zeros_like(v) for g, v in grads_and_vars]
            self.v = [tlx.zeros_like(v) for g, v in grads_and_vars]
        
        for i, (g, p) in enumerate(grads_and_vars):
            if g is None:
                continue
            
            
            if self.weight_decay != 0:
                g = g + self.weight_decay * p

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            
            update = step_size * (self.m[i] / (tlx.sqrt(self.v[i]) + self.epsilon))
            
            p.data -= update
            
            
            if hasattr(p, 'grad') and p.grad is not None:
                p.grad = None



np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                    formatter=dict(float=lambda x: "%9.3e" % x))

# ========== Training settings
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--seed', type=int, default=11, help='Random seed.')
parser.add_argument('-v', '--dev', type=int, default=0, help='Device id.')
parser.add_argument('-c', '--config', type=str, default='cora', help='Config file name.')
parser.add_argument('-m', '--algo', type=str, default=None, help='Model name')
parser.add_argument('-n', '--suffix', type=str, default='', help='Save name suffix.')
parser.add_argument('-a', '--thr_a', type=float, default=None, help='Threshold of adj.')
parser.add_argument('-w', '--thr_w', type=float, default=None, help='Threshold of weight.')
parser.add_argument('-l', '--layer', type=int, default=None, help='Layer.')
args = prepare_opt(parser)


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
                storage='state_gpu')
stopwatch = metric.Stopwatch()

# ========== download data
adj, feat, labels, idx, nfeat, nclass = load_edgelist(
    datastr=args.data, datapath=args.path,
    inductive=args.inductive, multil=args.multil, seed=args.seed
)

if args.algo.split('_')[0] in ['gcn2']:
    model = models.SandwitchThr(
        nlayer=args.layer, nfeat=nfeat, nhidden=args.hidden, nclass=nclass,
        thr_a=args.thr_a, thr_w=args.thr_w, dropout=args.dropout, layer=args.algo
    )
elif args.algo.split('_')[0] == 'mlp':
    model = models.MLP(
        nlayer=args.layer, nfeat=nfeat, nhidden=args.hidden, nclass=nclass,
        thr_w=args.thr_w, dropout=args.dropout, layer='mlp'
    )
else:
    model = models.GNNThr(
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

optimizer = PureTLXAdam(lr=args.lr, weight_decay=args.weight_decay)
loss_fn = tlx.losses.sigmoid_cross_entropy if args.multil else tlx.losses.softmax_cross_entropy_with_logits

def train(x, edge_idx, y, idx_split, epoch, verbose=False):
    model.train()
    if epoch < args.epochs // 2:
        model.set_scheme('pruneall', 'pruneall')
    else:
        model.set_scheme('pruneall', 'pruneinc')

    stopwatch.reset()
    stopwatch.start()

   
    output = model(x, edge_idx, node_lock=tlx.convert_to_tensor([]))[idx_split]
    loss = loss_fn(output, y)

   
    grads = optimizer.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

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