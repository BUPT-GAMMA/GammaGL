from typing import Union, Tuple, Optional
import geoopt
import numpy as np
import tensorlayerx as tlx
from gammagl.utils.manifold_math import sin_div
from geoopt.manifolds.stereographic.math import geodesic
from gammagl.mpops import unsorted_segment_sum


def get_eps(dtype):
    dtype_str = str(dtype)
    if "float64" in dtype_str or "float16" not in dtype_str and "64" in dtype_str:
        return 1e-7
    return 1e-4


def _to_tensor(val, dtype):
    if isinstance(val, (int, float)):
        return tlx.convert_to_tensor(val, dtype=dtype)
    return val


class Sphere(geoopt.Sphere):
    def __init__(self, learnable=False):
        super(Sphere, self).__init__()
        self.learnable = learnable
        self.k = tlx.nn.Parameter(tlx.convert_to_tensor([1.0]))

    def set_train(self, mode=True):
        pass

    def eval(self):
        pass

    def proju(self, x, u):
        return u - tlx.reduce_sum(x * u, axis=-1, keepdims=True) * x

    def origin(self, *size, dtype=None, device=None, seed=42):
        # Support origin(*shape), origin(shape_tuple), and origin(TensorShape).
        if len(size) == 1:
            s0 = size[0]
            if isinstance(s0, (tuple, list)):
                size = tuple(s0)
            elif hasattr(s0, "as_list"):
                size = tuple(s0.as_list())
            elif hasattr(s0, "__iter__") and not isinstance(s0, (int, np.integer)):
                size = tuple(list(s0))
        if len(size) == 0:
            size = (1,)

        size = tuple(int(d) for d in size)
        last_dim = int(size[-1])
        prefix_shape = tuple(size[:-1])

        if last_dim <= 1:
            return -tlx.ones(size, dtype=dtype)

        first = -tlx.ones(prefix_shape + (1,), dtype=dtype)
        rest = tlx.zeros(prefix_shape + (last_dim - 1,), dtype=dtype)
        return tlx.concat([first, rest], axis=-1)

    def cinner(self, x, y):
        # Point-wise inner product path to avoid accidental O(N^2) pairwise matrices.
        if tuple(x.shape) == tuple(y.shape):
            return tlx.reduce_sum(x * y, axis=-1, keepdims=True)
        return tlx.matmul(x, tlx.transpose(y, perm=list(range(y.ndim - 2)) + [y.ndim - 1, y.ndim - 2]))

    def inner(self, x, u, v=None, *, keepdim=False):
        if v is None:
            v = u
        inner = tlx.reduce_sum(u * v, axis=-1, keepdims=keepdim)
        return inner

    def geodesic(self, t, x, y):
        return geodesic(t, x, y, k=self.k)

    def Frechet_mean(self, x, weights=None, dim=0, keepdim=False, sum_idx=None):
        if sum_idx is not None:
            idx_dtype = tlx.int64 if tlx.BACKEND == "torch" else tlx.int32
            sum_idx = tlx.cast(sum_idx, idx_dtype)
            num_segments = int(tlx.convert_to_numpy(tlx.reduce_max(sum_idx))) + 1
        if weights is None:
            z = tlx.reduce_sum(x, axis=dim, keepdims=keepdim) if sum_idx is None \
                else unsorted_segment_sum(x, sum_idx, num_segments)
        else:
            z = tlx.reduce_sum(x * weights, axis=dim, keepdims=keepdim) if sum_idx is None \
                else unsorted_segment_sum(x * weights, sum_idx, num_segments)
        denorm = self.inner(None, z, keepdim=keepdim)
        denorm = tlx.sqrt(tlx.maximum(tlx.abs(denorm), _to_tensor(1e-8, denorm.dtype)))
        z = 1. / tlx.sqrt(self.k) * z / denorm
        return z

    def expmap0(self, u, dim=-1):
        pole = tlx.zeros_like(u)
        pole = tlx.concat([
            tlx.ones_like(pole[..., :1]) * (-1),
            pole[..., 1:]
        ], axis=-1)
        return self.expmap(pole, u)

    def expmap(self, x, u):
        norm_u = tlx.sqrt(tlx.reduce_sum(u * u, axis=-1, keepdims=True))
        exp = x * tlx.cos(norm_u) + u * sin_div(norm_u)
        retr = self.projx(x + u)
        cond = norm_u > _to_tensor(get_eps(norm_u.dtype), norm_u.dtype)
        retr = self.projx(x + u)
        return tlx.where(cond, exp, retr)

    def logmap0(self, y):
        x = self.origin(*y.shape, dtype=y.dtype, device=y.device)
        u = self.proju(x, y - x)
        dist = self.dist(x, y, keepdim=True)
        cond = dist > _to_tensor(get_eps(x.dtype), x.dtype)
        result = tlx.where(
            cond, u * dist / tlx.maximum(tlx.sqrt(tlx.reduce_sum(u * u, axis=-1, keepdims=True)), _to_tensor(get_eps(x.dtype), x.dtype)), u
        )
        return result

    def proju0(self, u):
        x = self.origin(*u.shape, dtype=u.dtype, device=u.device)
        u = u - tlx.reduce_sum(x * u, axis=-1, keepdims=True) * x
        return self._project_on_subspace(u)

    def projx(self, x):
        return x / tlx.maximum(tlx.sqrt(tlx.reduce_sum(x * x, axis=-1, keepdims=True)), _to_tensor(get_eps(x.dtype), x.dtype))

    def norm(self, u, x=None, *, keepdim=False):
        return tlx.sqrt(tlx.reduce_sum(u * u, axis=-1, keepdims=keepdim))

    def random_normal(
            self, *size, mean=0, std=1, dtype=None, device=None
    ):
        tens = tlx.convert_to_tensor(np.random.normal(size=size).astype(np.float32)) * std + mean
        return self.expmap0(tens)

    def transp0back(self, x, u):
        o = self.origin(*x.shape, dtype=x.dtype, device=x.device)
        return self.transp(x, o, u)

    def transp(self, x, y, u):
        u_proj = self.proju(x, u)
        return self.proju(y, u_proj)

    def dist(self, x, y, *, keepdim=False):
        same_shape = tuple(x.shape) == tuple(y.shape)
        eps = _to_tensor(get_eps(x.dtype), x.dtype)
        lower = -1.0 + eps
        upper = 1.0 - eps
        cosine = self.cinner(x, y) / self.k
        clipped = tlx.minimum(tlx.maximum(cosine, lower), upper)
        d = tlx.sqrt(self.k) * tlx.acos(clipped)
        if keepdim:
            return d
        return tlx.squeeze(d, axis=-1) if same_shape else d

    def logmap(self, x, y):
        u = self.proju(x, y - x)
        dist_xy = self.dist(x, y, keepdim=True)
        norm_u = tlx.sqrt(tlx.maximum(tlx.reduce_sum(u * u, axis=-1, keepdims=True), _to_tensor(get_eps(u.dtype), u.dtype)))
        return u * dist_xy / norm_u
