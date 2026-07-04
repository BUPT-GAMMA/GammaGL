import math
import numpy as np
from typing import Union, Tuple, Optional

import geoopt
import geoopt.manifolds.lorentz.math as lmath
from geoopt.manifolds.stereographic.math import geodesic

import tensorlayerx as tlx
from gammagl.mpops import unsorted_segment_sum
from gammagl.utils.manifold_math import sinh_div, arcosh, cosh, sinh, sin_div
from gammagl.utils.softmax import segment_softmax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_eps(dtype):
    dtype_str = str(dtype)
    if "float64" in dtype_str or ("float16" not in dtype_str and "64" in dtype_str):
        return 1e-7
    return 1e-4


def _to_tensor(val, dtype):
    if isinstance(val, (int, float)):
        return tlx.convert_to_tensor(val, dtype=dtype)
    return val


def _calculate_target_batch_dim(*dims):
    return max(dims) - 1


# ---------------------------------------------------------------------------
# Manifold: Euclidean
# ---------------------------------------------------------------------------

class Euclidean(geoopt.manifolds.Euclidean):
    def __init__(self):
        super().__init__()
        self.k = 1.0

    def set_train(self, mode=True):
        pass

    def eval(self):
        pass

    def random_normal(self, *shape, std=1.0):
        return tlx.random_normal(shape=shape, stddev=std)

    def expmap0(self, v):
        return v

    def logmap0(self, v):
        return v

    def proju0(self, v):
        return v

    def dist(self, x, u, *, keepdim=False):
        return tlx.sqrt(tlx.reduce_sum((x - u) * (x - u), axis=-1, keepdims=keepdim))

    def inner(self, x, u, v=None, *, keepdim=False):
        if v is None:
            v = u
        return tlx.reduce_sum(u * v, axis=-1, keepdims=keepdim)

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
        denorm = tlx.sqrt(tlx.maximum(tlx.abs(denorm), tlx.convert_to_tensor(1e-8)))
        z = z / denorm
        return z

    def norm(self, u, x=None, *, keepdim=False):
        return tlx.sqrt(tlx.reduce_sum(u * u, axis=-1, keepdims=keepdim))


# ---------------------------------------------------------------------------
# Manifold: ProductSpace
# ---------------------------------------------------------------------------

class ProductSpace(geoopt.ProductManifold):
    def __init__(self, *manifolds_with_shape):
        super(ProductSpace, self).__init__(*manifolds_with_shape)

    def set_train(self, mode=True):
        pass

    def eval(self):
        pass

    def logmap0(self, x):
        target_batch_dim = x.ndim - 1
        logmapped_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            logmapped = manifold.logmap0(point)
            logmapped = tlx.reshape(logmapped, (*logmapped.shape[:target_batch_dim], -1))
            logmapped_tensors.append(logmapped)
        return tlx.concat(logmapped_tensors, axis=-1)

    def proju0(self, u):
        target_batch_dim = u.ndim - 1
        projected = []
        for i, manifold in enumerate(self.manifolds):
            tangent = self.take_submanifold_value(u, i)
            proj = manifold.proju0(tangent)
            proj = tlx.reshape(proj, (*proj.shape[:target_batch_dim], -1))
            projected.append(proj)
        return tlx.concat(projected, axis=-1)

    def random_normal(self, *size, mean=0, std=1, dtype=None, device=None):
        shape = geoopt.utils.size2shape(*size)
        batch_shape = shape[:-1]
        points = []
        for manifold, shape in zip(self.manifolds, self.shapes):
            points.append(
                manifold.random_normal(batch_shape + shape, mean=mean, std=std, dtype=dtype, device=device)
            )
        tensor = self.pack_point(*points)
        return tensor

    def Frechet_mean(self, x, weights=None, dim=0, keepdim=False, sum_idx=None):
        target_batch_dim = x.ndim - 1
        mid_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            mid = manifold.Frechet_mean(point, weights, dim, keepdim, sum_idx)
            mid = tlx.reshape(mid, (*mid.shape[:target_batch_dim], -1))
            mid_tensors.append(mid)
        return tlx.concat(mid_tensors, axis=-1)


# ---------------------------------------------------------------------------
# Manifold: Sphere
# ---------------------------------------------------------------------------

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

    def random_normal(self, *size, mean=0, std=1, dtype=None, device=None):
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


# ---------------------------------------------------------------------------
# Manifold: Lorentz
# ---------------------------------------------------------------------------

class Lorentz(geoopt.Lorentz):
    def __init__(self, k=1.0, learnable=False):
        super(Lorentz, self).__init__(k, learnable)

    def set_train(self, mode=True):
        pass

    def eval(self):
        pass

    def expmap0(self, u, *, dim=-1):
        return self.expmap(self.origin(u.shape, dtype=u.dtype, device=u.device), u, dim=dim)

    def random_normal(self, *shape, std=1.0):
        return tlx.random_normal(shape=shape, stddev=std)

    def origin(self, shape=None, dtype=tlx.float32, device=None):
        sqrtK = tlx.sqrt(self.k)
        if shape is None or len(shape) == 0:
            origin = tlx.concat([sqrtK, tlx.zeros((self.ndim,), dtype=dtype)], axis=0)
        else:
            last_dim = shape[-1] if isinstance(shape[-1], int) else self.ndim + 1
            prefix_shape = shape[:-1] if len(shape) > 1 else ()
            origin = tlx.zeros(shape, dtype=dtype)
            ones = tlx.ones(list(prefix_shape) + [1], dtype=dtype)
            zeros_part = tlx.zeros(list(prefix_shape) + [last_dim - 1], dtype=dtype)
            origin = tlx.concat([sqrtK * ones, zeros_part], axis=-1)
        return origin

    def proju(self, x, u, *, dim=-1):
        x_sp = tlx.concat([-x[..., :1], x[..., 1:]], axis=dim)
        return u + tlx.reduce_sum(u * x_sp, axis=dim, keepdims=True) * x / tlx.sqrt(self.k)

    def projx(self, x, *, dim=-1):
        sqrtK = tlx.sqrt(self.k)
        x_sp = x * sqrtK
        x_sp = tlx.concat([-x_sp[..., :1], x_sp[..., 1:]], axis=dim)
        scale = sqrtK / tlx.sqrt(-tlx.reduce_sum(x_sp * x_sp, axis=dim, keepdims=True))
        return x * scale

    def norm(self, u, *, keepdim=False, dim=-1):
        x_sp = tlx.concat([-u[..., :1], u[..., 1:]], axis=dim)
        norm = tlx.sqrt(
            tlx.maximum(
                -tlx.reduce_sum(u * x_sp, axis=dim, keepdims=keepdim),
                _to_tensor(1e-8, u.dtype)
            )
        )
        return norm

    def inner(self, x, u, v=None, *, keepdim=False, dim=-1):
        if v is None:
            v = u
        x_sp = tlx.concat([-u[..., :1], u[..., 1:]], axis=dim)
        return tlx.reduce_sum(x_sp * v, axis=dim, keepdims=keepdim)

    def cinner(self, x, y):
        if tuple(x.shape) == tuple(y.shape):
            return tlx.reduce_sum((-x[..., :1] * y[..., :1]) + (x[..., 1:] * y[..., 1:]), axis=-1, keepdims=True)

        x = tlx.concat([
            -x[..., :1],
            x[..., 1:]
        ], axis=-1)
        return tlx.matmul(x, tlx.transpose(y, perm=list(range(y.ndim - 2)) + [y.ndim - 1, y.ndim - 2]))

    def geodesic(self, t, x, y):
        k_sqrt = tlx.sqrt(self.k)
        nomin = arcosh(-self.inner(None, x / k_sqrt, y / k_sqrt))
        v = self.logmap(x, y)
        return cosh(nomin * t) * x + k_sqrt * sinh(nomin * t) * v / self.norm(v, keepdim=True)

    def expmap(self, x, u, *, norm_tan=False, project=False, dim=-1):
        nomin = self.norm(u, keepdim=True, dim=dim)
        p = (
                cosh(nomin / tlx.sqrt(self.k)) * x
                + sinh_div(nomin / tlx.sqrt(self.k)) * u
        )
        return p

    def to_poincare(self, x, dim=-1):
        dn = x.shape[dim] - 1
        return x[..., 1:] / (x[..., :1] + tlx.sqrt(self.k))

    def from_poincare(self, x, dim=-1, eps=1e-6):
        x_norm_square = tlx.reduce_sum(x * x, axis=dim, keepdims=True)
        res = (
                tlx.sqrt(self.k)
                * tlx.concat((1 + x_norm_square, 2 * x), axis=dim)
                / (1.0 - x_norm_square + eps)
        )
        return res

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

    def dist(self, x, y, *, keepdim=False, dim=-1):
        if tuple(x.shape) == tuple(y.shape):
            inner_xy = tlx.reduce_sum((-x[..., :1] * y[..., :1]) + (x[..., 1:] * y[..., 1:]), axis=dim, keepdims=True)
            arg = -inner_xy / self.k
            arg = tlx.where(tlx.is_nan(arg), tlx.ones_like(arg) * (1.0 + 1e-5), arg)
            arg = tlx.maximum(arg, tlx.ones_like(arg) * (1.0 + 1e-5))
            d = tlx.sqrt(self.k) * arcosh(arg)
            return d if keepdim else tlx.squeeze(d, axis=dim)

        cinner_xy = self.cinner(x, y)
        arg = -cinner_xy / self.k
        arg = tlx.where(tlx.is_nan(arg), tlx.ones_like(arg) * (1.0 + 1e-5), arg)
        arg = tlx.maximum(arg, tlx.ones_like(arg) * (1.0 + 1e-5))
        return tlx.sqrt(self.k) * arcosh(arg)

    def logmap(self, x, y, *, dim=-1):
        u = self.proju(x, y - x, dim=dim)
        dist_xy = self.dist(x, y, keepdim=True, dim=dim)
        sqrtK = tlx.sqrt(self.k)
        denom = dist_xy * tlx.sinh(dist_xy / sqrtK)
        return sqrtK * denom * (self.logmap0(y, dim=dim) - self.logmap0(x, dim=dim))

    def transp0back(self, x, u, *, dim=-1):
        sqrtK = tlx.sqrt(self.k)
        o = self.origin(x.shape, dtype=x.dtype, device=x.device)
        xo = self.proju(o, x, dim=dim)
        inner_xu = self.inner(o, xo, u, keepdim=True, dim=dim)
        inner_xx = self.inner(o, xo, xo, keepdim=True, dim=dim)
        eps = _to_tensor(1e-8, inner_xx.dtype)
        return u - 2 * inner_xu / (inner_xx + eps) * xo

    def proju0(self, v, *, dim=-1):
        o = self.origin(v.shape, dtype=v.dtype, device=v.device)
        return self.proju(o, v, dim=dim)

    def logmap0(self, x, *, dim=-1):
        d = x.shape[-1] - 1
        K = self.k
        sqrtK = K ** 0.5
        y = x[..., 1:]
        y = tlx.reshape(y, [-1, d])
        y_norm = tlx.sqrt(tlx.reduce_sum(y * y, axis=-1, keepdims=True)) + _to_tensor(1e-8, y.dtype)
        theta = tlx.expand_dims(x[..., 0] / sqrtK, axis=-1)
        theta = tlx.maximum(theta, _to_tensor(1.0 + get_eps(x.dtype), x.dtype))
        res = sqrtK * arcosh(theta) * y / y_norm
        zeros = tlx.zeros([tlx.get_tensor_shape(x)[0], 1], dtype=x.dtype)
        res = tlx.concat([zeros, res], axis=-1)
        return res


# ---------------------------------------------------------------------------
# RGT Layers
# ---------------------------------------------------------------------------

class EuclideanEncoder(tlx.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bias=True, activation=tlx.relu, dropout=0.1):
        super().__init__()
        self.lin = tlx.layers.Linear(in_features=in_dim, out_features=hidden_dim, b_init=tlx.initializers.constant(1.0) if bias else None)
        self.activation = activation
        self.proj = tlx.layers.Linear(in_features=hidden_dim, out_features=out_dim, b_init=tlx.initializers.constant(1.0) if bias else None)
        self.dropout = tlx.nn.Dropout(p=dropout)
        self.drop = dropout

    def forward(self, x):
        x = self.lin(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.proj(self.dropout(x))
        x = x / (tlx.sqrt(tlx.reduce_sum(x * x, axis=-1, keepdims=True)) + 1e-8)
        return x


class ManifoldEncoder(tlx.nn.Module):
    def __init__(self, manifold, in_dim, hidden_dim, out_dim, bias=True, activation=None, dropout=0.1):
        super().__init__()
        self.manifold = manifold
        self.lin = ConstCurveLinear(manifold, in_dim, out_dim, bias=bias, dropout=dropout, activation=activation)
        self.agg = ConstCurveAgg(manifold, out_dim, dropout, use_att=False)

    def forward(self, x, edge_index):
        x = self.manifold.expmap0(x)
        x = self.lin(x)
        x = self.agg(x, edge_index)
        return x


class ConstCurveLinear(tlx.nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.0,
                 scale=10,
                 fixscale=False,
                 activation=None):
        super().__init__()
        self.manifold = manifold
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = tlx.layers.Linear(
            in_features=self.in_features, out_features=self.out_features, b_init=tlx.initializers.constant(1.0) if bias else None)
        self.reset_parameters()
        self.dropout = tlx.nn.Dropout(p=dropout)
        self.scale = tlx.nn.Parameter(tlx.ones((1,)) * math.log(scale))
        self.sign = -1. if isinstance(manifold, Lorentz) else 1.

    def forward(self, x):
        if self.activation is not None:
            x = self.activation(x)
        x = self.dropout(x)
        x = self.weight(x)
        x_narrow = x[..., 1:]
        time = tlx.sigmoid(x[..., :1]) * tlx.exp(self.scale) + 1.1 if isinstance(self.manifold, Lorentz)  \
        else tlx.sigmoid(x[..., :1]) - 0.5
        scale = self.sign * (1. / self.manifold.k - time * time) / \
            tlx.maximum(tlx.reduce_sum(x_narrow * x_narrow, axis=-1, keepdims=True), tlx.convert_to_tensor(1e-8))
        x = tlx.concat([time, x_narrow * tlx.sqrt(scale)], axis=-1)
        return x

    def reset_parameters(self):
        pass


class ConstCurveAgg(tlx.nn.Module):
    def __init__(self, manifold, in_features, dropout=0.0, use_att=False):
        super(ConstCurveAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_features
        self.dropout = dropout
        self.use_att = use_att
        if self.use_att:
            self.key_linear = ConstCurveLinear(manifold, in_features, in_features)
            self.query_linear = ConstCurveLinear(manifold, in_features, in_features)
            self.bias = tlx.nn.Parameter(tlx.zeros((1,)) + 20)
            self.scale = tlx.nn.Parameter(tlx.zeros((1,)) + math.sqrt(in_features))
        if isinstance(manifold, Lorentz):
            self.neg_dist = lambda x, y: 2 + 2 * manifold.cinner(x, y)
        else:
            self.neg_dist = lambda x, y: -manifold.dist(x, y) ** 2
        self.sign = -1. if isinstance(manifold, Lorentz) else 1.

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        src = tlx.cast(tlx.convert_to_tensor(src), tlx.int64)
        dst = tlx.cast(tlx.convert_to_tensor(dst), tlx.int64)
        if self.use_att:
            query = self.query_linear(x)
            key = self.key_linear(x)
            att_adj = 2 + 2 * self.manifold.cinner(tlx.gather(query, dst), tlx.gather(key, src))
            att_adj = att_adj / self.scale + self.bias
            att_adj = tlx.sigmoid(att_adj)
            support_t = unsorted_segment_sum(att_adj * tlx.gather(x, dst), src, x.shape[0])
        else:
            support_t = unsorted_segment_sum(tlx.gather(x, dst), src, x.shape[0])

        denorm = self.sign * self.manifold.inner(None, support_t, keepdim=True)

        denorm = tlx.sqrt(tlx.maximum(tlx.abs(denorm), tlx.convert_to_tensor(1e-8)))
        output = 1. / tlx.sqrt(self.manifold.k) * support_t / denorm
        return output
