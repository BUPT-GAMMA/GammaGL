import os
import gc
import random
import argparse
import numpy as np
import ptflops

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from utils.logger import Logger, ModelLogger, prepare_opt
from utils.loader import load_embedding
import utils.metric as metric
from archs import flops_modules_dict
import archs.models as models


np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                    formatter=dict(float=lambda x: "% 9.3e" % x))
torch.set_printoptions(linewidth=160, edgeitems=5)

# ========== Training settings
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--seed', type=int, default=11, help='Random seed.')
parser.add_argument('-v', '--dev', type=int, default=1, help='Device id.')
parser.add_argument('-c', '--config', type=str, default='./config/cora_mb.json', help='Config file name.')
parser.add_argument('-m', '--algo', type=str, default=None, help='Model name')
parser.add_argument('-n', '--suffix', type=str, default='', help='Save name suffix.')
parser.add_argument('-a', '--thr_a', type=float, default=None, help='Threshold of adj.')
parser.add_argument('-w', '--thr_w', type=float, default=None, help='Threshold of weight.')
parser.add_argument('-l', '--layer', type=int, default=None, help='Layer.')
parser.add_argument('-p', '--hop', type=int, default=None, help='Hop.')
args = prepare_opt(parser)

num_thread = 0 if args.data in ['cora', 'citeseer', 'pubmed'] else 8
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.dev >= 0:
    with torch.cuda.device(args.dev):
        torch.cuda.manual_seed(args.seed)

if not ('_'  in args.algo):
    args.thr_a, args.thr_w = 0.0, 0.0
args.chn['delta'] = args.thr_a
args.chn['hop'] = args.hop if isinstance(args.hop, int) else args.chn['hop']
flag_run = f"{args.seed}-{args.thr_a:.1e}-{args.thr_w:.1e}"
logger = Logger(args.data, args.algo, flag_run=flag_run)
logger.save_opt(args)
model_logger = ModelLogger(logger, patience=args.patience, cmp='max',
                           prefix='model'+args.suffix, storage='state_gpu')
stopwatch = metric.Stopwatch()

# ========== Load
feat, labels, idx, nfeat, nclass, macs_pre, time_pre = load_embedding(datastr=args.data,
                datapath=args.path, algo=args.algo, algo_chn=args.chn,
                inductive=args.inductive, multil=args.multil, seed=args.seed)

model = models.MLP(nlayer=args.layer, nfeat=nfeat, nhidden=args.hidden, nclass=nclass,
                   thr_w=args.thr_w, dropout=args.dropout, layer=args.algo,)
model.reset_parameters()
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

ds_train = Data.TensorDataset(feat['train'], labels['train'])
loader_train = Data.DataLoader(dataset=ds_train, batch_size=args.batch,
                               shuffle=True, num_workers=num_thread)
ds_val = Data.TensorDataset(feat['val'], labels['val'])
loader_val = Data.DataLoader(dataset=ds_val, batch_size=args.batch,
                             shuffle=False, num_workers=num_thread)
ds_test = Data.TensorDataset(feat['test'], labels['test'])
loader_test = Data.DataLoader(dataset=ds_test, batch_size=args.batch,
                              shuffle=False, num_workers=num_thread)


def train(epoch, ld=loader_train, verbose=False):
    model.train()

    loss_list = []
    stopwatch.reset()
    for it, (x, y) in enumerate(ld):
        x, y = x.cuda(args.dev), y.cuda(args.dev)

        if it == 0:
            if epoch < args.epochs//2:
                model.set_scheme('pruneall')
            else:
                model.set_scheme('pruneinc')
        else:
            model.set_scheme('keep')

        stopwatch.start()
        optimizer.zero_grad()
        output = model(x)
        loss_batch = loss_fn(output, y)
        loss_batch.backward()
        optimizer.step()
        stopwatch.pause()

        loss_list.append(loss_batch.item())

    with torch.cuda.device(args.dev):
        torch.cuda.empty_cache()
    gc.collect()
    return np.mean(loss_list), stopwatch.time


def eval(ld, verbose=False):
    model.eval()
    model.set_scheme('keep')
    output_l, labels_l = None, None
    calc = metric.F1Calculator(nclass)
    stopwatch.reset()

    with torch.no_grad():
        for _, (x, y) in enumerate(ld):
            x, y = x.cuda(args.dev), y.cuda(args.dev)

            stopwatch.start()
            output = model(x)
            stopwatch.pause()

            output = output.detach()
            if args.multil:
                output = torch.where(output > 0, torch.tensor(1, device=output.device), torch.tensor(0, device=output.device))
            else:
                output = output.argmax(dim=1)
            calc.update(y, output)

            # output = output.cpu().detach().numpy()
            # y = y.cpu().detach().numpy()
            # output_l = output if output_l is None else np.concatenate((output_l, output), axis=0)
            # labels_l = y if labels_l is None else np.concatenate((labels_l, y), axis=0)
    if args.multil:
        res = calc.compute('micro')
    else:
        res = calc.compute('micro')
    return res, stopwatch.time, output_l, labels_l


def cal_flops(ld, verbose=False):
    model.eval()
    model.set_scheme('keep')

    macs, nparam = ptflops.get_model_complexity_info(model, (nfeat,),
                        custom_modules_hooks=flops_modules_dict,
                        as_strings=False, print_per_layer_stat=verbose, verbose=verbose)
    return macs/1e9, nparam/1e3


# ========== Train
# print('-' * 20, flush=True)
with torch.cuda.device(args.dev):
    torch.cuda.empty_cache()
time_tol, macs_tol = metric.Accumulator(), metric.Accumulator()
epoch_conv, acc_best = 0, 0

for epoch in range(1, args.epochs+1):
    verbose = epoch % 1 == 0 and (logger.lvl_log > 0)
    loss_train, time_epoch = train(ld=loader_train,
                                   epoch=epoch, verbose=verbose)
    time_tol.update(time_epoch)
    acc_val, _, _, _ = eval(ld=loader_val)
    scheduler.step(acc_val)
    macs_epoch, _ = cal_flops(ld=loader_val)
    macs_tol.update(macs_epoch*len(idx['train']))

    if verbose:
        res = f"Epoch:{epoch:04d} | train loss:{loss_train:.4f}, val acc:{acc_val:.4f}, time:{time_tol.val:.4f}, macs:{macs_tol.val:.4f}"
        if logger.lvl_log > 1:
            logger.print(res)

    # Log convergence
    acc_best = model_logger.save_best(acc_val, epoch=epoch)
    if model_logger.is_early_stop(epoch=epoch):
        pass
    #     break     # >> Enable to early stop
    else:
        epoch_conv = max(0, epoch - model_logger.patience)

# ========== Test
# print('-' * 20, flush=True)
model = model_logger.load()
if args.dev >= 0:
    model = model.cuda(args.dev)
with torch.cuda.device(args.dev):
    torch.cuda.empty_cache()

acc_test, time_test, outl, labl = eval(ld=loader_test)
# mem_ram, mem_cuda = metric.get_ram(), metric.get_cuda_mem(args.dev)
# num_param, mem_param = metric.get_num_params(model), metric.get_mem_params(model)
macs_test, _ = cal_flops(ld=loader_val)
macs_test *= len(idx['test'])

n = len(idx['train']) + len(idx['val']) + len(idx['test'])
r_train, r_test = len(idx['train'])/n, len(idx['test'])/n
macs_wtr, macs_wte = macs_tol.val, macs_test
macs_tol.update(macs_pre * r_train, 0)
macs_test += macs_pre * r_test
numel_a = macs_pre * 1e6 / nfeat
numel_w = model.get_numel()

# ========== Log
if logger.lvl_config > 0:
    print(f"[Val] best acc: {acc_best:0.5f} (epoch: {epoch_conv}/{epoch}), [Test] best acc: {acc_test:0.5f}", flush=True)
if logger.lvl_config > 1:
    print(f"[Pre]   time: {time_pre:0.4f} s, MACs: {macs_pre:0.4f} G")
    print(f"[Train] time: {time_tol.val:0.4f} s (avg: {time_tol.avg*1000:0.1f} ms), MACs: {macs_tol.val:0.3f} G (avg: {macs_tol.avg:0.1f} G)")
    print(f"[Test]  time: {time_test:0.4f} s, MACs: {macs_test:0.4f} G, Num adj: {numel_a:0.3f} k, Num weight: {numel_w:0.3f} k")
    # print(f"RAM: {mem_ram:.3f} GB, CUDA: {mem_cuda:.3f} GB, Num params: {num_param:0.4f} M, Mem params: {mem_param:0.4f} MB")
    print(f"Train MACs: {macs_wtr:0.4f} G, Pre MACs: {macs_pre:0.4f} G (avg: {macs_pre*1e6/n:0.4f} K), Test MACs: {macs_wte:0.4f} G (avg: {macs_wte*1e6/len(idx['test']):0.4f} K)")
if logger.lvl_config > 2:
    logger_tab = Logger(args.data, args.algo, flag_run=flag_run, dir=('./save', args.data))
    logger_tab.file_log = logger_tab.path_join('log_mb.csv')
    hstr, cstr = logger_tab.str_csvg(data=args.data, algo=args.algo, seed=args.seed, thr_a=args.thr_a, thr_w=args.thr_w,
                                    acc_test=acc_test, conv_epoch=epoch_conv, epoch=epoch,
                                    time_train=time_tol.val, macs_train=macs_tol.val,
                                    macs_a=macs_pre, macs_wtr=macs_wtr, macs_wte=macs_wte,
                                    time_test=time_test, macs_test=macs_test, numel_a=numel_a, numel_w=numel_w,
                                    hop=args.chn['hop'], layer=args.layer, time_pre=time_pre)
    logger_tab.print_header(hstr, cstr)
