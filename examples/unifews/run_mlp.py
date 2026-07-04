import gc
import random
import argparse
import numpy as np

try:
    import ptflops
    HAS_PTFLOPS = True
except ImportError:
    HAS_PTFLOPS = False

import tensorlayerx as tlx
from tensorlayerx import nn
from tensorlayerx.dataflow import Dataset, DataLoader
from tensorlayerx.model import WithLoss, TrainOneStep

import sys
sys.path.append("/root/GammaGL")
from gammagl.utils.logger_unifews import Logger, ModelLogger
from examples.unifews.loader import load_embedding
import gammagl.utils.metric_unifews as metric
from gammagl.models.gnn_unifews import flops_modules_dict
import gammagl.models.mlp_unifews as models

np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                    formatter=dict(float=lambda x: "%.6e" % x))

# ========== Training settings ==========
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--seed', type=int, default=11)
parser.add_argument('-v', '--dev', type=int, default=0)
parser.add_argument('-m', '--algo', type=str, default='sgc')
parser.add_argument('-n', '--suffix', type=str, default='')
parser.add_argument('-a', '--thr_a', type=float, default=0.0001)
parser.add_argument('-w', '--thr_w', type=float, default=0.1)
parser.add_argument('-l', '--layer', type=int, default=2)
parser.add_argument('-p', '--hop', type=int, default=2)
parser.add_argument('--data', type=str, default='cora')
parser.add_argument('--path', type=str, default='./data/')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--hidden', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--inductive', action='store_true', default=False)
parser.add_argument('--multil', action='store_true', default=False)
args = parser.parse_args()

# Construct chn config (previously loaded from JSON)
args.chn = {'hop': args.hop, 'delta': args.thr_a, 'alpha': 0.0, 'rrz': 0.5}

num_thread = 0 if args.data in ['cora', 'citeseer', 'pubmed'] else 8
random.seed(args.seed)
np.random.seed(args.seed)
tlx.set_seed(args.seed)

if '_' not in args.algo: args.thr_a, args.thr_w = 0.0, 0.0
args.chn['delta'] = args.thr_a
args.chn['hop'] = args.hop if isinstance(args.hop, int) else args.chn['hop']
flag_run = f"{args.seed}-{args.thr_a:.1e}-{args.thr_w:.1e}"

logger = Logger(args.data, args.algo, flag_run=flag_run)
logger.save_opt(args)
model_logger = ModelLogger(logger, patience=args.patience, cmp='max',
                           prefix=f'model{args.suffix}', storage='state')
stopwatch = metric.Stopwatch()

# ========== Data Load ==========
feat, labels, idx, nfeat, nclass, macs_pre, time_pre = load_embedding(
    datastr=args.data, datapath=args.path, algo=args.algo, algo_chn=args.chn,
    inductive=args.inductive, multil=args.multil, seed=args.seed
)

# ========== Model Init ==========
model = models.MLP_unifews(
    nlayer=args.layer, nfeat=nfeat, nhidden=args.hidden, nclass=nclass,
    thr_w=args.thr_w, dropout=args.dropout, layer=args.algo,
)

# Trigger model build with a dummy forward pass
dummy_x = tlx.convert_to_tensor(np.random.randn(1, nfeat), dtype=tlx.float32)
_ = model(dummy_x, edge_idx=None)
model.reset_parameters()

if logger.lvl_config > 1:
    print(type(model).__name__, args.algo, args.thr_a, args.thr_w)
model_logger.register(model, save_init=False)

# ========== Training Components ==========
loss_fn = tlx.losses.sigmoid_cross_entropy_with_logits if args.multil else tlx.losses.softmax_cross_entropy_with_logits
net_with_loss = WithLoss(model, loss_fn)

train_weights = model.trainable_weights

optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
train_one_step = TrainOneStep(net_with_loss, optimizer, train_weights)

best_val_acc = 0.0
patience_counter = 0
lr_factor = 0.5
lr_patience = 15

# ========== Dataset ==========
class TensorDataset(Dataset):
    def __init__(self, data, label):
        self.data = tlx.convert_to_tensor(data, dtype=tlx.float32)
        self.label = tlx.convert_to_tensor(label, dtype=tlx.float32 if args.multil else tlx.int64)
    def __getitem__(self, idx): return self.data[idx], self.label[idx]
    def __len__(self): return len(self.data)

loader_train = DataLoader(TensorDataset(feat['train'], labels['train']), batch_size=args.batch, shuffle=True, num_workers=num_thread)
loader_val   = DataLoader(TensorDataset(feat['val'], labels['val']), batch_size=args.batch, shuffle=False, num_workers=num_thread)
loader_test  = DataLoader(TensorDataset(feat['test'], labels['test']), batch_size=args.batch, shuffle=False, num_workers=num_thread)

# ========== Train Func ==========
def train(epoch, ld=loader_train, verbose=False):
    model.train()
    loss_list = []
    stopwatch.reset()

    for it, (x, y) in enumerate(ld):
        if it == 0:
            model.set_scheme('pruneall' if epoch < args.epochs // 2 else 'pruneinc', 'pruneall')
        else:
            model.set_scheme('keep', 'keep')

        stopwatch.start()
        loss_batch = train_one_step(x, y)
        stopwatch.pause()
        loss_list.append(float(loss_batch))

    gc.collect()
    return np.mean(loss_list), stopwatch.time

# ========== Eval Func ==========
def eval(ld, verbose=False):
    model.eval()
    model.set_scheme('keep', 'keep')
    calc = metric.F1Calculator(nclass)
    stopwatch.reset()

    for x, y in ld:
        stopwatch.start()
        output = model(x, edge_idx=None)
        stopwatch.pause()

        if args.multil:
            output = tlx.where(output > 0, 1, 0)
        else:
            output = tlx.argmax(output, axis=1)

        calc.update(y, output)

    return calc.compute('micro'), stopwatch.time, None, None

# ========== FLOPs Func ==========
def cal_flops(ld, verbose=False):
    if not HAS_PTFLOPS:
        return 0.0, 0.0
    model.eval()
    model.set_scheme('keep', 'keep')
    macs, nparam = ptflops.get_model_complexity_info(
        model, (nfeat,), custom_modules_hooks=flops_modules_dict,
        as_strings=False, print_per_layer_stat=verbose, verbose=verbose
    )
    return macs / 1e9, nparam / 1e3

# ========== Train Loop ==========
time_tol, macs_tol = metric.Accumulator(), metric.Accumulator()
epoch_conv, acc_best = 0, 0

for epoch in range(1, args.epochs + 1):
    verbose = (epoch % 1 == 0) and (logger.lvl_log > 0)
    loss_train, time_epoch = train(epoch, ld=loader_train, verbose=verbose)
    time_tol.update(time_epoch)

    acc_val, _, _, _ = eval(loader_val)

    if acc_val > best_val_acc:
        best_val_acc = acc_val
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= lr_patience:
            optimizer.lr *= lr_factor
            patience_counter = 0
            if logger.lvl_log > 1:
                logger.print(f"Reducing learning rate to {optimizer.lr:.6f}")

    macs_epoch, _ = cal_flops(loader_val)
    macs_tol.update(macs_epoch * len(idx['train']))

    if verbose and logger.lvl_config > 1:
        res = f"Epoch:{epoch:04d} | train loss:{loss_train:.4f}, val acc:{acc_val:.4f}, time:{time_tol.val:.4f}, macs:{macs_tol.val:.4f}"
        logger.print(res)

    acc_best = model_logger.save_best(acc_val, epoch)
    if not model_logger.is_early_stop(epoch):
        epoch_conv = epoch - model_logger.patience

acc_test, time_test, _, _ = eval(loader_test)
macs_test, _ = cal_flops(loader_val)
macs_test *= len(idx['test'])

# Final metrics
n = len(idx['train']) + len(idx['val']) + len(idx['test'])
r_train, r_test = len(idx['train']) / n, len(idx['test']) / n
macs_wtr, macs_wte = macs_tol.val, macs_test
macs_tol.update(macs_pre * r_train, 0)
macs_test += macs_pre * r_test
numel_a = macs_pre * 1e6 / nfeat

def get_numel_safe(m):
    try:
        res = m.get_numel()
        return res[1] if isinstance(res, tuple) else res
    except:
        if hasattr(m, 'trainable_weights'):
            return sum(np.prod(p.shape) for p in m.trainable_weights) / 1000.0
        return 0.0

numel_w = get_numel_safe(model)

# ========== Logging ==========
if logger.lvl_config > 0:
    print(f"[Val] best acc: {acc_best:.5f} (epoch: {epoch_conv}/{epoch}), [Test] acc: {acc_test:.5f}", flush=True)
if logger.lvl_config > 1:
    print(f"[Pre]   time: {time_pre:.4f} s, MACs: {macs_pre:.4f} G")
    print(f"[Train] time: {time_tol.val:.4f} s, avg: {time_tol.avg*100:.1f} ms, MACs: {macs_tol.val:.3f} G, avg: {macs_tol.avg:.1f} G")
    print(f"[Test]  time: {time_test:.4f} s, MACs: {macs_test:.4f} G, Num adj: {numel_a:.3f} k, Num weight: {numel_w:.3f} k")
    print(f"Train MACs: {macs_wtr:.4f} G, Pre MACs: {macs_pre:.4f} G, Test MACs: {macs_wte:.4f} G")
if logger.lvl_config > 2:
  
    import os
    save_dir = './save'
    os.makedirs(save_dir, exist_ok=True)  

    logger_tab = Logger(args.data, args.algo, flag_run=flag_run, dir=save_dir)
    logger_tab.file_log = os.path.join(save_dir, f'log_mb_{flag_run}.csv')
    
    hstr, cstr = logger_tab.str_csvg(
        data=args.data, algo=args.algo, seed=args.seed, thr_a=args.thr_a, thr_w=args.thr_w,
        acc_test=acc_test, conv_epoch=epoch_conv, epoch=epoch, time_train=time_tol.val,
        macs_train=macs_tol.val, macs_a=macs_pre, macs_wtr=macs_wtr, macs_wte=macs_wte,
        time_test=time_test, macs_test=macs_test, numel_a=numel_a, numel_w=numel_w,
        hop=args.chn['hop'], layer=args.layer, time_pre=time_pre
    )
    logger_tab.print_header(hstr, cstr)
    