import os
import random
import argparse
import numpy as np
import ptflops

import torch
import torch.nn as nn
import torch.optim as optim

from utils.logger import Logger, ModelLogger, prepare_opt
from utils.loader import load_edgelist
import utils.metric as metric
from archs import identity_n_norm, flops_modules_dict
import archs.models as models


np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                    formatter=dict(float=lambda x: "% 9.3e" % x))
torch.set_printoptions(linewidth=160, edgeitems=5)

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

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.dev >= 0:
    with torch.cuda.device(args.dev):
        torch.cuda.manual_seed(args.seed)

if not ('_'  in args.algo):
    args.thr_a, args.thr_w = 0.0, 0.0
flag_run = f"{args.seed}-{args.thr_a:.1e}-{args.thr_w:.1e}"
logger = Logger(args.data, args.algo, flag_run=flag_run)
logger.save_opt(args)
model_logger = ModelLogger(logger,
                patience=args.patience,
                cmp='max',
                prefix='model'+args.suffix,
                storage='state_ram' if args.data in ['cs', 'physics', 'arxiv'] else 'state_gpu')
stopwatch = metric.Stopwatch()

# ========== Load
adj, feat, labels, idx, nfeat, nclass = load_edgelist(datastr=args.data, datapath=args.path,
                inductive=args.inductive, multil=args.multil, seed=args.seed)

if args.algo.split('_')[0] in ['gcn2']:
    model = models.SandwitchThr(nlayer=args.layer, nfeat=nfeat, nhidden=args.hidden, nclass=nclass,
                        thr_a=args.thr_a, thr_w=args.thr_w, dropout=args.dropout, layer=args.algo)
elif args.algo.split('_')[0] in ['mlp']:
    model = models.MLP(nlayer=args.layer, nfeat=nfeat, nhidden=args.hidden, nclass=nclass,
                        thr_w=args.thr_w, dropout=args.dropout, layer='mlp')
else:
    model = models.GNNThr(nlayer=args.layer, nfeat=nfeat, nhidden=args.hidden, nclass=nclass,
                        thr_a=args.thr_a, thr_w=args.thr_w, dropout=args.dropout, layer=args.algo)
model.reset_parameters()
model.kwargs['diag'] = None
# diag = model.kwargs['diag'] if ('_' in args.algo) else None
diag = model.kwargs['diag']
adj['train'] = identity_n_norm(adj['train'], edge_weight=None, num_nodes=feat['train'].shape[0],
                    rnorm=model.kwargs['rnorm'], diag=diag)
if logger.lvl_config > 1:
    print(type(model).__name__, args.algo, args.thr_a, args.thr_w)
if logger.lvl_config > 2:
    print(model)
model_logger.register(model, save_init=False)
if args.dev >= 0:
    model = model.cuda(args.dev)

# ========== Train helper
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, threshold=1e-4, patience=15, verbose=False)
loss_fn = nn.BCEWithLogitsLoss() if args.multil else nn.CrossEntropyLoss()


def train(x, edge_idx, y, idx_split, epoch, verbose=False):
    model.train()
    if epoch < args.epochs//2:
        model.set_scheme('pruneall', 'pruneall')
    else:
        # model.set_scheme('pruneinc', 'pruneinc')
        model.set_scheme('pruneall', 'pruneinc')
    x, y = x.cuda(args.dev), y.cuda(args.dev)
    if isinstance(edge_idx, tuple):
        edge_idx = (edge_idx[0].cuda(args.dev), edge_idx[1].cuda(args.dev))
    else:
        edge_idx = edge_idx.cuda(args.dev)
    stopwatch.reset()

    stopwatch.start()
    optimizer.zero_grad()
    output = model(x, edge_idx, node_lock=torch.Tensor([]), verbose=verbose)[idx_split]
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    stopwatch.pause()

    return loss.item(), stopwatch.time


def eval(x, edge_idx, y, idx_split, verbose=False):
    model.eval()
    model.set_scheme('keep', 'keep')
    # model.set_scheme('full', 'keep')
    x, y = x.cuda(args.dev), y.cuda(args.dev)
    if isinstance(edge_idx, tuple):
        edge_idx = (edge_idx[0].cuda(args.dev), edge_idx[1].cuda(args.dev))
    else:
        edge_idx = edge_idx.cuda(args.dev)
    calc = metric.F1Calculator(nclass)
    stopwatch.reset()

    with torch.no_grad():
        stopwatch.start()
        output = model(x, edge_idx, node_lock=idx_split, verbose=verbose)[idx_split]
        # output = model(x, edge_idx, node_lock=torch.Tensor([]), verbose=verbose)[idx_split]
        stopwatch.pause()

        output = output.cpu().detach()
        ylabel = y.cpu().detach()
        if args.multil:
            output = torch.where(output > 0, torch.tensor(1, device=output.device), torch.tensor(0, device=output.device))
        else:
            output = output.argmax(dim=1)
        calc.update(ylabel, output)

        output = output.numpy()
        ylabel = ylabel.numpy()

    res = calc.compute(('macro' if args.multil else 'micro'))
    return res, stopwatch.time, output, y


def cal_flops(x, edge_idx, idx_split, verbose=False):
    model.eval()
    model.set_scheme('keep', 'keep')
    x = x.cuda(args.dev)
    if isinstance(edge_idx, tuple):
        edge_idx = (edge_idx[0].cuda(args.dev), edge_idx[1].cuda(args.dev))
    else:
        edge_idx = edge_idx.cuda(args.dev)

    handle = model.register_forward_hook(models.GNNThr.batch_counter_hook)
    model.__batch_counter_handle__ = handle
    macs, nparam = ptflops.get_model_complexity_info(model, (1,1,1),
                        input_constructor=lambda _: {'x': x, 'edge_idx': edge_idx},
                        custom_modules_hooks=flops_modules_dict,
                        as_strings=False, print_per_layer_stat=verbose, verbose=verbose)
    return macs/1e9


# ========== Train
# print('-' * 20, flush=True)
with torch.cuda.device(args.dev):
    torch.cuda.empty_cache()
time_tol, macs_tol = metric.Accumulator(), metric.Accumulator()
epoch_conv, acc_best = 0, 0

for epoch in range(1, args.epochs+1):
    verbose = epoch % 1 == 0 and (logger.lvl_log > 0)
    loss_train, time_epoch = train(x=feat['train'], edge_idx=adj['train'],
                                   y=labels['train'], idx_split=idx['train'],
                                   epoch=epoch, verbose=verbose)
    time_tol.update(time_epoch)
    acc_val, _, _, _ = eval(x=feat['train'], edge_idx=adj['train'],
                            y=labels['val'], idx_split=idx['val'])
    scheduler.step(acc_val)
    macs_epoch = cal_flops(x=feat['train'], edge_idx=adj['train'], idx_split=idx['train'])
    macs_tol.update(macs_epoch)

    if verbose:
        res = f"Epoch:{epoch:04d} | train loss:{loss_train:.4f}, val acc:{acc_val:.4f}, time:{time_tol.val:.4f}, macs:{macs_tol.val:.4f}"
        if logger.lvl_log > 1:
            logger.print(res)

    # Log convergence
    acc_best = model_logger.save_best(acc_val, epoch=epoch)
    if model_logger.is_early_stop(epoch=epoch):
        pass
        # break     # Enable to early stop
    else:
        epoch_conv = max(0, epoch - model_logger.patience)

# ========== Test
# print('-' * 20, flush=True)
model = model_logger.load('best')
if args.dev >= 0:
    model = model.cuda(args.dev)
with torch.cuda.device(args.dev):
    torch.cuda.empty_cache()

adj['test'] = identity_n_norm(adj['test'], edge_weight=None, num_nodes=feat['test'].shape[0],
                    rnorm=model.kwargs['rnorm'], diag=model.kwargs['diag'])
acc_test, time_test, outl, labl = eval(x=feat['test'], edge_idx=adj['test'],
                                       y=labels['test'], idx_split=idx['test'])
# mem_ram, mem_cuda = metric.get_ram(), metric.get_cuda_mem(args.dev)
# num_param, mem_param = metric.get_num_params(model), metric.get_mem_params(model)
macs_test = cal_flops(x=feat['test'], edge_idx=adj['test'], idx_split=idx['test'])
numel_a, numel_w = model.get_numel()

# ========== Log
if logger.lvl_config > 0:
    print(f"[Val] best acc: {acc_best:0.5f} (epoch: {epoch_conv}/{epoch}), [Test] best acc: {acc_test:0.5f}", flush=True)
if logger.lvl_config > 0:
    print(f"[Train] time: {time_tol.val:0.4f} s (avg: {time_tol.avg*1000:0.1f} ms), MACs: {macs_tol.val:0.3f} G (avg: {macs_tol.avg:0.1f} G)")
    print(f"[Test]  time: {time_test:0.4f} s, MACs: {macs_test:0.4f} G, Num adj: {numel_a:0.3f} k, Num weight: {numel_w:0.3f} k")
    # print(f"RAM: {mem_ram:.3f} GB, CUDA: {mem_cuda:.3f} GB, Num params: {num_param:0.4f} M, Mem params: {mem_param:0.4f} MB")
if logger.lvl_config > 2:
    logger_tab = Logger(args.data, args.algo, flag_run=flag_run, dir=('./save', args.data))
    logger_tab.file_log = logger_tab.path_join('log.csv')
    hstr, cstr = logger_tab.str_csv(data=args.data, algo=args.algo, seed=args.seed, thr_a=args.thr_a, thr_w=args.thr_w,
                                    acc_test=acc_test, conv_epoch=epoch_conv, epoch=epoch,
                                    time_train=time_tol.val, macs_train=macs_tol.val,
                                    time_test=time_test, macs_test=macs_test, numel_a=numel_a, numel_w=numel_w)
    logger_tab.print_header(hstr, cstr)
