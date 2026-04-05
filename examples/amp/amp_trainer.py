#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
AMP on ZINC. ``gammagl.datasets.ZINC`` **固定**从 ``<dataset_path>/ZINC/raw/`` 读数据（与类名一致），
不能直接把数据根设为 ``ZINC-PE`` 文件夹名。

若你只有 ``ZINC-PE/raw/*.index``：仍需 ``train/val/test.pickle``（来自官方 ``molecules.zip``）。
可用 ``--zinc_local_zip`` / 网络下载补齐后，再用 ``--zinc_index_dir .../ZINC-PE/raw`` 覆盖三份 ``*.index``。
"""

import os

# Must be set before ``import tensorlayerx`` (otherwise TLX may try TensorFlow backend).
os.environ.setdefault("TL_BACKEND", "torch")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import os.path as osp
import random
import shutil
import ssl
import sys
import urllib.request
import zipfile
from typing import Optional

import numpy as np
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss


def _patch_torch_and_tlx_named_members() -> None:
    """
    修复 ``TypeError: _named_members() got an unexpected keyword argument 'remove_duplicate'``。

    **根因**：PyTorch 2.x 的 ``named_parameters`` 会向 ``_named_members`` 传入 ``remove_duplicate``，
    但 TensorLayerX 的 ``tensorlayerx.nn.core.core_torch.Module`` **重写了** ``_named_members``，
    且签名里没有该参数；方法解析时走到 TLX 的实现而非 ``torch.nn.Module``，故只补丁后者无效。

    做法：包装 TLX 的 ``Module._named_members``，吞掉 ``remove_duplicate`` 再调用原实现；并顺带包装
    ``torch.nn.Module._named_members`` 以防纯 ``nn.Module`` 混搭旧实现。
    """
    import torch.nn as nn

    def _wrap_nn_module_named_members() -> None:
        orig = nn.Module.__dict__.get("_named_members", nn.Module._named_members)
        if getattr(orig, "__amp_remove_dup_patched__", False):
            return

        def _wrapped(self, *args, **kwargs):
            kwargs.pop("remove_duplicate", None)
            return orig(self, *args, **kwargs)

        _wrapped.__amp_remove_dup_patched__ = True  # type: ignore[attr-defined]
        nn.Module._named_members = _wrapped  # type: ignore[method-assign]

    def _wrap_tlx_module_named_members() -> None:
        from tensorlayerx.nn.core import core_torch

        orig = core_torch.Module._named_members
        if getattr(orig, "__amp_remove_dup_patched__", False):
            return

        def _wrapped(self, get_members_fn, prefix="", recurse=True, remove_duplicate=True, **kwargs):
            kwargs.pop("remove_duplicate", None)
            return orig(self, get_members_fn, prefix=prefix, recurse=recurse)

        _wrapped.__amp_remove_dup_patched__ = True  # type: ignore[attr-defined]
        core_torch.Module._named_members = _wrapped  # type: ignore[method-assign]

    _wrap_nn_module_named_members()
    _wrap_tlx_module_named_members()


_patch_torch_and_tlx_named_members()


def _patch_torch_optim_functional_adam() -> None:
    """
    兼容 TensorLayerX 与 PyTorch 2.x 的 ``torch.optim._functional.adam``：

    - 补上关键字 ``maximize=False``；
    - TLX 传入的 ``state_steps`` 为 Python int 列表，2.x 要求「单元素张量」列表，在此转换。
    """
    import inspect

    import torch
    import torch.optim._functional as F

    orig = getattr(F, "adam", None)
    if orig is None or getattr(orig, "__amp_adam_maximize_patched__", False):
        return
    try:
        if "maximize" not in inspect.signature(orig).parameters:
            return
    except (ValueError, TypeError):
        return

    def _adam(*args, **kwargs):
        kwargs.setdefault("maximize", False)
        a = list(args)
        # F.adam(..., state_steps, *, amsgrad=...) — state_steps 为第 6 个位置参数（0-based 下标 5）
        if len(a) >= 6:
            steps = a[5]
            if steps and not isinstance(steps[0], torch.Tensor):
                dev = a[0][0].device if a[0] else None
                a[5] = [torch.tensor(s, device=dev, dtype=torch.int64) for s in steps]
            args = tuple(a)
        return orig(*args, **kwargs)

    _adam.__amp_adam_maximize_patched__ = True  # type: ignore[attr-defined]
    F.adam = _adam  # type: ignore[assignment]


_patch_torch_optim_functional_adam()

# ``AMPModel`` 依赖 ``global_add_pool`` 别名（部分 GammaGL 版本仅有 ``global_sum_pool``）
def _patch_glob_add_pool_alias() -> None:
    import gammagl.layers.pool.glob as g

    if not hasattr(g, "global_add_pool") and hasattr(g, "global_sum_pool"):
        g.global_add_pool = g.global_sum_pool


_patch_glob_add_pool_alias()

from gammagl.datasets import ZINC
from gammagl.loader import DataLoader
from gammagl.models import AMPModel, amp_elbo_regression_loss

# ---------------------------------------------------------------------------
# ZINC raw layout (must match ``gammagl.datasets.zinc.ZINC``)
# ---------------------------------------------------------------------------
_ZINC_URL = "https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1"
_SPLIT_URL_TMPL = (
    "https://raw.githubusercontent.com/graphdeeplearning/"
    "benchmarking-gnns/master/data/molecules/{}.index"
)
_RAW_NAMES = (
    "train.pickle",
    "val.pickle",
    "test.pickle",
    "train.index",
    "val.index",
    "test.index",
)


def _download_timeout() -> Optional[float]:
    v = os.environ.get("AMP_ZINC_DOWNLOAD_TIMEOUT", "300").strip()
    if v in ("", "0", "none", "None", "inf"):
        return None
    return float(v)


def _urlopen(url: str, timeout: Optional[float]):
    context = ssl._create_unverified_context()
    proxies = urllib.request.getproxies()
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(proxies),
        urllib.request.HTTPHandler(),
        urllib.request.HTTPSHandler(context=context),
    )
    return opener.open(url, timeout=timeout)


def _download_file(url: str, dest_path: str, timeout: Optional[float]) -> None:
    os.makedirs(osp.dirname(dest_path) or ".", exist_ok=True)
    print(f"[amp_trainer] Downloading -> {dest_path}\n  {url}", file=sys.stderr)
    resp = _urlopen(url, timeout=timeout)
    try:
        with open(dest_path, "wb") as f:
            while True:
                chunk = resp.read(10 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    finally:
        resp.close()


def ensure_zinc_raw(
    dataset_root: str,
    *,
    molecules_url: Optional[str] = None,
    split_url_tmpl: Optional[str] = None,
    local_zip: Optional[str] = None,
    github_proxy: bool = False,
) -> None:
    """Populate ``<root>/ZINC/raw`` so GammaGL's ``ZINC`` skips its own download."""
    root = osp.abspath(osp.normpath(dataset_root))
    raw_dir = osp.join(root, "ZINC", "raw")
    if all(osp.isfile(osp.join(raw_dir, n)) for n in _RAW_NAMES):
        print(f"[amp_trainer] ZINC raw already complete under {raw_dir}", file=sys.stderr)
        return

    os.makedirs(root, exist_ok=True)
    if osp.isdir(raw_dir):
        shutil.rmtree(raw_dir)

    timeout = _download_timeout()
    mol_url = (molecules_url or _ZINC_URL).strip()
    split_tmpl = (split_url_tmpl or _SPLIT_URL_TMPL).strip()

    zip_path = osp.join(root, "molecules.zip")
    if local_zip:
        lz = osp.abspath(osp.expanduser(local_zip))
        if not osp.isfile(lz):
            raise FileNotFoundError(f"--zinc_local_zip not a file: {lz}")
        shutil.copy2(lz, zip_path)
        print(f"[amp_trainer] Using local zip: {lz}", file=sys.stderr)
    else:
        _download_file(mol_url, zip_path, timeout)

    print(f"[amp_trainer] Extracting {zip_path}", file=sys.stderr)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    os.remove(zip_path)

    molecules_dir = osp.join(root, "molecules")
    if not osp.isdir(molecules_dir):
        raise RuntimeError(
            f"Expected folder 'molecules' after extracting ZINC zip, not found under {root}"
        )
    os.makedirs(osp.join(root, "ZINC"), exist_ok=True)
    shutil.move(molecules_dir, raw_dir)

    for split in ("train", "val", "test"):
        u = split_tmpl.format(split)
        if github_proxy or os.environ.get("GGL_GITHUB_PROXY", "").upper() == "TRUE":
            u = "https://ghproxy.com/" + u
        dest = osp.join(raw_dir, f"{split}.index")
        _download_file(u, dest, timeout)

    missing = [n for n in _RAW_NAMES if not osp.isfile(osp.join(raw_dir, n))]
    if missing:
        raise RuntimeError(f"ZINC raw incomplete, missing: {missing}")


def _copy_zinc_split_indices(dataset_root: str, index_src_dir: str) -> bool:
    """
    将外部目录（如 ``ZINC-PE/raw``）下的 ``*.index`` 复制到 ``<dataset_path>/ZINC/raw/``。
    返回是否复制了任一文件（用于触发重新 process）。
    """
    if not (index_src_dir or "").strip():
        return False
    src = osp.abspath(osp.expanduser(index_src_dir.strip()))
    if not osp.isdir(src):
        raise FileNotFoundError(f"--zinc_index_dir 不是目录: {src}")
    raw_dir = osp.join(osp.abspath(osp.normpath(dataset_root)), "ZINC", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    copied = False
    for name in ("train.index", "val.index", "test.index"):
        sp = osp.join(src, name)
        if osp.isfile(sp):
            shutil.copy2(sp, osp.join(raw_dir, name))
            copied = True
            print(f"[amp_trainer] 已覆盖索引: {name} <- {sp}", file=sys.stderr)
    if not copied:
        print(f"[amp_trainer] 在 {src} 未找到 train/val/test.index，跳过覆盖。", file=sys.stderr)
    return copied


class AMPELBOLoss(WithLoss):
    def __init__(self, net, num_atom_types, n_obs):
        super().__init__(backbone=net, loss_fn=None)
        self.num_atom_types = int(num_atom_types)
        self.n_obs = float(n_obs)
        self.onehot = tlx.nn.OneHot(depth=self.num_atom_types)

    def forward(self, data, label):
        x_idx = tlx.cast(data.x, tlx.int64)
        if len(tlx.get_tensor_shape(x_idx)) == 2 and tlx.get_tensor_shape(x_idx)[1] == 1:
            x_idx = tlx.squeeze(x_idx, axis=1)
        x = self.onehot(x_idx)

        y = tlx.cast(label, tlx.float32)
        if len(tlx.get_tensor_shape(y)) == 1:
            y = tlx.expand_dims(y, axis=-1)

        # 训练需 ELBO 辅助量；``AMPModel.forward`` 只返回标量预测，须调用 ``forward_elbo``
        _, output_stack, aux = self.backbone_network.forward_elbo(
            x, data.edge_index, None, data.batch
        )
        log_h, log_o, log_l, entropy_qL, qL_b = aux
        return amp_elbo_regression_loss(
            output_state=output_stack,
            targets=y,
            log_p_theta_hidden=log_h,
            log_p_theta_output=log_o,
            log_p_L=log_l,
            entropy_qL=entropy_qL,
            qL_probs=qL_b,
            n_obs=self.n_obs,
        )


def main(args):
    fix_seed(args.seed)

    if not (args.dataset_path or "").strip():
        raise SystemExit("Please set --dataset_path to the ZINC root directory.")

    root = osp.abspath(osp.normpath(args.dataset_path.strip()))
    local_zip = (args.zinc_local_zip or "").strip() or None
    if not local_zip:
        auto_zip = osp.join(root, "ZINC", "molecules.zip")
        if osp.isfile(auto_zip):
            local_zip = auto_zip
            print(f"[amp_trainer] 自动使用 {local_zip}", file=sys.stderr)

    ensure_zinc_raw(
        args.dataset_path,
        molecules_url=args.zinc_molecules_url or None,
        split_url_tmpl=args.zinc_split_url_template or None,
        local_zip=local_zip,
        github_proxy=args.zinc_github_proxy,
    )

    indices_updated = _copy_zinc_split_indices(args.dataset_path, args.zinc_index_dir)
    zinc_reload = bool(indices_updated or args.zinc_force_reload)

    train_dataset = ZINC(
        args.dataset_path, subset=True, split="train", force_reload=zinc_reload
    )
    val_dataset = ZINC(args.dataset_path, subset=True, split="val", force_reload=zinc_reload)
    test_dataset = ZINC(args.dataset_path, subset=True, split="test", force_reload=zinc_reload)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    net = AMPModel(
        in_channels=args.num_atom_types,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        max_depth=args.max_depth,
        theta_prior_scale=args.theta_prior_scale,
        folded_loc_init=args.folded_loc,
        folded_scale_init=args.folded_scale,
        global_aggregation=True,
        filter_messages=args.filter_messages,
        name="AMP",
    )

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
    train_weights = net.trainable_weights
    loss_func = AMPELBOLoss(net, num_atom_types=args.num_atom_types, n_obs=len(train_dataset))
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    best_val = float("inf")
    best_test = float("inf")

    for epoch in range(1, args.n_epoch + 1):
        net.set_train()
        epoch_loss = 0.0
        n_g = 0
        for data in train_loader:
            loss = train_one_step(data, data.y)
            epoch_loss += float(loss) * int(data.num_graphs)
            n_g += int(data.num_graphs)

        train_loss = epoch_loss / max(n_g, 1)
        val_mae = evaluate_mae(net, val_loader, args.num_atom_types)
        test_mae = evaluate_mae(net, test_loader, args.num_atom_types)
        if val_mae < best_val:
            best_val = val_mae
            best_test = test_mae

        if epoch % args.display_step == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:0>4d}]  train_loss: {train_loss:.6f}  "
                f"val_mae: {val_mae:.4f}  test_mae: {test_mae:.4f}  "
                f"best_val: {best_val:.4f}  best_test: {best_test:.4f}"
            )

    print(f"Done. Best val MAE={best_val:.4f}, best test MAE={best_test:.4f}")


def evaluate_mae(model, loader, num_atom_types):
    model.set_eval()
    onehot = tlx.nn.OneHot(depth=int(num_atom_types))
    total = 0.0
    n_graphs = 0
    for data in loader:
        x_idx = tlx.cast(data.x, tlx.int64)
        if len(tlx.get_tensor_shape(x_idx)) == 2 and tlx.get_tensor_shape(x_idx)[1] == 1:
            x_idx = tlx.squeeze(x_idx, axis=1)
        x = onehot(x_idx)
        pred = model(x, data.edge_index, None, data.batch)
        y = tlx.cast(data.y, tlx.float32)
        if len(tlx.get_tensor_shape(y)) == 1:
            y = tlx.expand_dims(y, axis=-1)
        mae = tlx.reduce_mean(tlx.abs(pred - y))
        total += float(mae) * int(data.num_graphs)
        n_graphs += int(data.num_graphs)
    return total / max(n_graphs, 1)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tlx.set_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMP on ZINC (GammaGL)")
    parser.add_argument("--dataset_path", type=str, default="", help="ZINC root directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id; <0 for CPU")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epoch", type=int, default=400)
    parser.add_argument("--display_step", type=int, default=10)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--num_atom_types", type=int, default=28)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--hidden_channels", type=int, default=64)

    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--theta_prior_scale", type=float, default=10.0)
    parser.add_argument("--folded_loc", type=float, default=5.0)
    parser.add_argument("--folded_scale", type=float, default=3.0)

    parser.add_argument(
        "--filter_messages",
        type=str,
        default="embedding-no-weight-sharing",
        help="embedding-no-weight-sharing | input_features | none",
    )

    parser.add_argument(
        "--zinc_local_zip",
        type=str,
        default="",
        help="Local path to molecules.zip (same as GammaGL Dropbox archive); skips remote zip download.",
    )
    parser.add_argument(
        "--zinc_molecules_url",
        type=str,
        default="",
        help="Override URL for molecules.zip (mirror).",
    )
    parser.add_argument(
        "--zinc_split_url_template",
        type=str,
        default="",
        help="Override index URL template with one '{}' for split name (train/val/test).",
    )
    parser.add_argument(
        "--zinc_github_proxy",
        action="store_true",
        help="Prefix GitHub raw URLs with https://ghproxy.com/ when downloading *.index.",
    )
    parser.add_argument(
        "--zinc_index_dir",
        type=str,
        default="",
        help=(
            "覆盖 split 索引的目录（内含 train.index / val.index / test.index），"
            "例如与脚本同级的 ZINC-PE/raw。仍会使用 <dataset_path>/ZINC/raw/ 下的 pickle。"
        ),
    )
    parser.add_argument(
        "--zinc_force_reload",
        action="store_true",
        help="强制 ZINC 重新 process（忽略已缓存的 subset/*/torch/*.pt）。",
    )

    args = parser.parse_args()
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    if args.filter_messages.lower() == "none":
        args.filter_messages = None

    main(args)
