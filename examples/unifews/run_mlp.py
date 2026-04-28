import os
os.environ['TL_BACKEND'] = 'torch'

import gc
import random
import argparse
import numpy as np
import ptflops

import tensorlayerx as tlx
from tensorlayerx import nn
from tensorlayerx.dataflow import Dataset, DataLoader
from tensorlayerx.model import WithLoss, TrainOneStep


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
# =====================================================================


# =====================================================================

from gammagl.utils.logger_gamma import Logger, ModelLogger, prepare_opt
from gammagl.utils.loader_gamma import load_embedding
import gammagl.utils.metric_gamma as metric
from gammagl.models.gnn_model import flops_modules_dict
import gammagl.models.mlp_model as models

np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                    formatter=dict(float=lambda x: "%.6e" % x))

# ========== Training settings ==========
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--seed', type=int, default=11)
parser.add_argument('-v', '--dev', type=int, default=1)
parser.add_argument('-c', '--config', type=str, default='./config/cora_mb.json')
parser.add_argument('-m', '--algo', type=str, default=None)
parser.add_argument('-n', '--suffix', type=str, default='')
parser.add_argument('-a', '--thr_a', type=float, default=None)
parser.add_argument('-w', '--thr_w', type=float, default=None)
parser.add_argument('-l', '--layer', type=int, default=None)
parser.add_argument('-p', '--hop', type=int, default=None)
args = prepare_opt(parser)

num_thread = 0 if args.data in ['cora', 'citeseer', 'pubmed'] else 8
random.seed(args.seed)
np.random.seed(args.seed)
tlx.set_seed(args.seed)
device = f'cuda:{args.dev}' if args.dev >= 0 else 'cpu'

if '_' not in args.algo: args.thr_a, args.thr_w = 0.0, 0.0
args.chn['delta'] = args.thr_a
args.chn['hop'] = args.hop if isinstance(args.hop, int) else args.chn['hop']
flag_run = f"{args.seed}-{args.thr_a:.1e}-{args.thr_w:.1e}"

logger = Logger(args.data, args.algo, flag_run=flag_run)
logger.save_opt(args)
model_logger = ModelLogger(logger, patience=args.patience, cmp='max',
                           prefix=f'model{args.suffix}', storage='state_gpu')
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

# 1. LAZY INIT FIX: Trigger model build using a CPU tensor first!
dummy_x_cpu = tlx.convert_to_tensor(np.random.randn(1, nfeat), dtype=tlx.float32)
_ = model(dummy_x_cpu, edge_idx=None)

# 2. Now that weights are built, move the model to the target device
if hasattr(model, 'to'):
    model.to(device)
model.reset_parameters()

if logger.lvl_config > 1:
    print(type(model).__name__, args.algo, args.thr_a, args.thr_w)
model_logger.register(model, save_init=False)

# ========== Training Components ==========
loss_fn = tlx.losses.sigmoid_cross_entropy_with_logits if args.multil else tlx.losses.softmax_cross_entropy_with_logits
net_with_loss = WithLoss(model, loss_fn)

train_weights = model.trainable_weights
if not train_weights and hasattr(model, 'parameters'):
    train_weights = [p for p in model.parameters() if p.requires_grad]

# Use our Custom TLX Optimizer and standard TrainOneStep
optimizer = PureTLXAdam(lr=args.lr, weight_decay=args.weight_decay)
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
        x = tlx.convert_to_tensor(x, device=device)
        y = tlx.convert_to_tensor(y, device=device)

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
def to_numpy(t):
    # Use the native TLX converter - it handles GPU->CPU and Detaching automatically
    return tlx.convert_to_numpy(t)

def eval(ld, verbose=False):
    model.eval()
    model.set_scheme('keep', 'keep')
    calc = metric.F1Calculator(nclass)
    stopwatch.reset()

    for x, y in ld:
        # 1. Move input data to the GPU for inference
        x = tlx.convert_to_tensor(x, device=device)
        y = tlx.convert_to_tensor(y, device=device)

        stopwatch.start()
        output = model(x, edge_idx=None)
        stopwatch.pause()

        # 2. Get predictions (still on GPU)
        if args.multil:
            output = tlx.where(output > 0, 1, 0)
        else:
            output = tlx.argmax(output, axis=1)
        
        # 3. FIX: Move to CPU but KEEP as Tensors
        # This allows metric_gamma.py to use tlx.reshape/tlx.cast without crashing,
        # and ensures it won't trigger the "cuda:1 to numpy" error later.
        y_cpu = tlx.convert_to_tensor(y, device='cpu')
        output_cpu = tlx.convert_to_tensor(output, device='cpu')
        
        calc.update(y_cpu, output_cpu)

    return calc.compute('micro'), stopwatch.time, None, None

# ========== FLOPs Func ==========
def cal_flops(ld, verbose=False):
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

# ========== Test ==========
model = model_logger.load()
if hasattr(model, 'to'):
    model.to(device)
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
    