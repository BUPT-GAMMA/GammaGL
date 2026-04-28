import os
os.environ['TL_BACKEND'] = 'torch'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from abc import ABC, abstractmethod

def safe_norm(x, p=2, axis=None, keepdims=False):
    x_abs = tlx.abs(x)
    if p == 1:
        return tlx.reduce_sum(x_abs, axis=axis, keepdims=keepdims)
    return tlx.sqrt(tlx.reduce_sum(x_abs * x_abs, axis=axis, keepdims=keepdims))

class BasePruningMethod(ABC):
    PRUNING_TYPE = 'unstructured'
    _tensor_name: str = None

    def __call__(self, module, inputs):
        if not hasattr(module, self._tensor_name + "_mask"):
            return
        self._cached_weight = self.apply_mask(module)

    @abstractmethod
    def compute_mask(self, t, default_mask):
        pass

    def apply_mask(self, module):
        if not hasattr(module, self._tensor_name + "_mask") or not hasattr(module, self._tensor_name + "_orig"):
            return getattr(module, self._tensor_name)
        mask = getattr(module, self._tensor_name + "_mask")
        orig = getattr(module, self._tensor_name + "_orig")
        return mask.to(dtype=orig.dtype) * orig

    @classmethod
    def apply(cls, module, name, *args, importance_scores=None, **kwargs):
        hooks_to_remove = []
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
                hooks_to_remove.append(k)
        for k in hooks_to_remove:
            del module._forward_pre_hooks[k]

        method = cls(*args, **kwargs)
        method._tensor_name = name

        orig = getattr(module, name)
        if importance_scores is None:
            importance_scores = orig

        if not hasattr(module, name + "_orig"):
            setattr(module, name + "_orig", orig.detach())
            default_mask = tlx.ones_like(orig)
        else:
            default_mask = getattr(module, name + "_mask")

        mask = method.compute_mask(importance_scores, default_mask)
        setattr(module, name + "_mask", mask)

        module.register_forward_pre_hook(method)
        return method

    def remove(self, module):
        if self._tensor_name:
            for suffix in ["_orig", "_mask"]:
                attr = self._tensor_name + suffix
                if hasattr(module, attr):
                    delattr(module, attr)

class RandomUnstructured(BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, amount) -> None:
        self._validate_amount(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        tensor_size = t.numel()
        nparams_toprune = self._get_prune_count(self.amount, tensor_size)
        nparams_toprune = max(0, min(nparams_toprune, tensor_size))
        

        mask = tlx.ones_like(default_mask)

        if nparams_toprune != 0:
            
            prob = tlx.random_uniform(shape=tlx.get_tensor_shape(t), dtype=t.dtype)
            flat_prob = tlx.reshape(prob, (-1,))
        
            _, topk_indices = tlx.topk(flat_prob, k=nparams_toprune, largest=False)
            flat_mask = tlx.reshape(mask, (-1,))
            
            flat_mask = tlx.scatter_update(flat_mask, topk_indices, tlx.zeros_like(topk_indices, dtype=flat_mask.dtype))
            mask = tlx.reshape(flat_mask, tlx.get_tensor_shape(default_mask))

        return mask

    def _validate_amount(self, amount):
        if isinstance(amount, float):
            if not (0.0 <= amount <= 1.0):
                raise ValueError("剪枝比例必须在 0.0 ~ 1.0 之间")
        elif isinstance(amount, int):
            if amount < 0:
                raise ValueError("剪枝数量不能为负数")
        else:
            raise TypeError("amount 必须为 int(绝对数量) 或 float(剪枝比例)")

    def _get_prune_count(self, amount, tensor_size):
        if isinstance(amount, int):
            return amount
        return int(amount * tensor_size)

class prune:
    @staticmethod
    def is_pruned(module: Module) -> bool:
        return any(key.endswith("_mask") for key in dir(module))

    @staticmethod
    def remove(module: Module, name: str):
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
                hook.remove(module)
                break

    RandomUnstructured = RandomUnstructured

def prune_threshold(x, threshold=1e-3):
    norm_vals = safe_norm(x, axis=1) / x.shape[1]
    idx_0 = norm_vals < threshold
    x = tlx.where(idx_0, tlx.zeros_like(x), x)
    return x, idx_0

def prune_topk(x, k=0.2):
    num_0 = int(x.shape[0] * k)
    x_norm = safe_norm(x, axis=1)
    _, idx_0 = tlx.topk(x_norm, num_0)
    mask = tlx.ones((x.shape[0],), dtype=tlx.bool)
    mask[idx_0] = False
    x = tlx.where(mask[:, None], x, tlx.zeros_like(x))
    return x, idx_0

def rewind(module: Module, name: str):
    orig_name = name + "_orig"
    mask_name = name + "_mask"
    if hasattr(module, orig_name):
        delattr(module, orig_name)
        delattr(module, mask_name)
        hooks_to_del = [k for k, h in module._forward_pre_hooks.items() if isinstance(h, BasePruningMethod) and h._tensor_name == name]
        for k in hooks_to_del:
            del module._forward_pre_hooks[k]

class ThrInPrune(BasePruningMethod):
    PRUNING_TYPE = 'structured'
    def __init__(self, threshold, dim=0):
        self.threshold = threshold
        self.dim = dim

    def compute_mask(self, t, default_mask):
        tmax = tlx.reduce_max(tlx.abs(t)) * (1 - 1e-3)
        threshold = tlx.where(self.threshold > tmax, tmax, self.threshold)
        mask = tlx.ones_like(default_mask)
        mask = tlx.where(tlx.abs(t) < threshold, 0.0, mask)
        return mask

class ThrProdPrune(BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, t, default_mask):
        mask = tlx.ones_like(default_mask)
        mask = tlx.where(tlx.abs(t) < self.threshold, 0.0, mask)
        return mask