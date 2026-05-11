import tensorlayerx as tlx
import geoopt
from gammagl.mpops import unsorted_segment_sum


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

    def inner(
        self, x, u, v=None, *, keepdim=False
    ):
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
