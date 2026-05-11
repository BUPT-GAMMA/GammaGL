import geoopt
import tensorlayerx as tlx
import geoopt.manifolds.lorentz.math as lmath
from gammagl.utils.manifold_math import sinh_div, arcosh, cosh, sinh
from gammagl.mpops import unsorted_segment_sum


def get_eps(dtype):
    dtype_str = str(dtype)
    if "float64" in dtype_str or ("float16" not in dtype_str and "64" in dtype_str):
        return 1e-7
    return 1e-4


def _to_tensor(val, dtype):
    if isinstance(val, (int, float)):
        return tlx.convert_to_tensor(val, dtype=dtype)
    return val


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
        # For same-shape inputs, return point-wise Lorentzian inner product to
        # avoid constructing an O(N^2) pairwise matrix.
        if tuple(x.shape) == tuple(y.shape):
            return tlx.reduce_sum((-x[..., :1] * y[..., :1]) + (x[..., 1:] * y[..., 1:]), axis=-1, keepdims=True)

        # Keep pairwise behavior for explicit pairwise-distance use cases.
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

    def expmap(
        self, x, u, *, norm_tan=False, project=False, dim=-1
    ):
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
            arg = tlx.where(tlx.isnan(arg), tlx.ones_like(arg) * (1.0 + 1e-5), arg)
            arg = tlx.maximum(arg, tlx.ones_like(arg) * (1.0 + 1e-5))
            d = tlx.sqrt(self.k) * arcosh(arg)
            return d if keepdim else tlx.squeeze(d, axis=dim)

        cinner_xy = self.cinner(x, y)
        arg = -cinner_xy / self.k
        arg = tlx.where(tlx.isnan(arg), tlx.ones_like(arg) * (1.0 + 1e-5), arg)
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
        # Use TLX concat instead of TF-specific concat
        zeros = tlx.zeros([tlx.get_tensor_shape(x)[0], 1], dtype=x.dtype)
        res = tlx.concat([zeros, res], axis=-1)
        return res
