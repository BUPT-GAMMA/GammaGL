import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn

from gammagl.utils import degree
from gammagl.layers.conv import MessagePassing


class MGNNI_m_iter(MessagePassing):
    r"""The mgnni operator from the `"Multiscale Graph Neural Networks with Implicit Layers"
    <https://arxiv.org/abs/2210.08353>`_ paper

    .. math::
       Z^{(l+1)} =\gamma g(F)Z^{(l)}S^{(m)}+f(X;G)

    where :math `\gamma` denotes the contraction factor,
    :math `m` denotes a hyperparameter for graph scale(i.e., the power of adjacency matrix) and
    :math `f(X;G)` is a parameterized transformation on input features and graphs,
    the normalized weight matrix :math:`g(F)` are computed as

    .. math::
       g(F) =\frac{1}{\|F^\top F\|_\text{F}+\epsilon_F}F^\top F

    Parameters
    ----------
    m: int
        Size of each input sample to
        derive the size from the first input(s) to the forward method.
    k: int
        The power of adjacency matrix.
        The greater the k, the further the distance to capture the information
    threshold: int
        Threshold for convergence.
        Convergence is considered when the difference
        between the two times is less than this threshold
    max_iter: int
        Maximum number of iterative solver iterations
    gamma: float
        The contraction factor.
        The smaller the gamma, the faster the contraction,
        the smaller the capture range; the larger the gamma,
        the larger the capture range, but it is difficult to converge and inefficient
    layer_norm: bool, optional
        whether to use layer norm. (default: :obj:`False`)

    """
    def __init__(self, m, k, threshold, max_iter, gamma, layer_norm=False):
        super(MGNNI_m_iter, self).__init__()
        self.F = nn.Parameter(tlx.convert_to_tensor(np.zeros((m, m)), dtype=tlx.float32))
        self.layer_norm = layer_norm
        self.gamma = gamma
        self.k = k
        self.max_iter = max_iter
        self.threshold = threshold
        self.f_solver = fwd_solver
        self.b_solver = fwd_solver

    def reset_parameters(self):
        initor = tlx.initializers.Zeros()
        self.F = self._get_weights("F", shape=self.F.shape, init=initor)

    def _inner_func(self, Z, X, edge_index, edge_weight, num_nodes):
        P = tlx.ops.transpose(Z)
        ei = tlx.ops.convert_to_tensor(edge_index)
        src, dst = ei[0], ei[1]
        if edge_weight is None:
            edge_weight = tlx.ones(shape=(ei.shape[1], 1))
        edge_weight = tlx.reshape(edge_weight, (-1,))

        deg = degree(src, num_nodes=num_nodes, dtype=tlx.float32)
        norm = tlx.pow(deg, -0.5)
        weights = tlx.ops.gather(norm, src) * tlx.reshape(edge_weight, (-1,))

        deg = degree(dst, num_nodes=num_nodes, dtype=tlx.float32)
        norm = tlx.pow(deg, -0.5)
        weights = tlx.reshape(weights, (-1,)) * tlx.ops.gather(norm, dst)

        for _ in range(self.k):
            P = self.propagate(P, ei, edge_weight=weights, num_nodes=num_nodes)

        Z = tlx.ops.transpose(P)

        Z_new = self.gamma * g(self.F) @ Z + X
        del Z, P, ei
        return Z_new

    def forward(self, X, edge_index, edge_weight, num_nodes):
        Z, abs_diff = self.f_solver(lambda Z: self._inner_func(Z, X, edge_index, edge_weight, num_nodes),
                                    z_init=tlx.zeros_like(X),
                                    threshold=self.threshold,
                                    max_iter=self.max_iter)
        Z = tlx.convert_to_tensor(Z)

        new_Z = Z

        if self.is_train:
            if tlx.BACKEND != 'paddle':
                new_Z = self._inner_func(tlx.Variable(Z, 'Z'), X, edge_index, edge_weight, num_nodes)
            else:
                Z.stop_gradient = False
                new_Z = self._inner_func(Z, X, edge_index, edge_weight, num_nodes)

        return new_Z


def fwd_solver(f, z_init, threshold, max_iter):
    z_prev, z = z_init, f(z_init)
    nstep = 0
    while nstep < max_iter:
        z_prev, z = z, f(z)
        abs_diff = tlx.ops.convert_to_numpy(norm(z_prev - z)).item()
        if abs_diff < threshold:
            break
        nstep += 1
        del z_prev
    if nstep == max_iter:
        print(f'step {nstep}, not converged, abs_diff: {abs_diff}')
    return z, abs_diff


def norm(input, p="fro", dim=None, keepdim=False, out=None, dtype=None):
    if p == "fro":
        norm_np = np.linalg.norm(tlx.convert_to_numpy(input), ord="fro", axis=dim, keepdims=keepdim)
    elif p == "nuc":
        norm_np = np.linalg.norm(tlx.convert_to_numpy(input), ord="nuc", axis=dim, keepdims=keepdim)
    else:
        norm_np = np.linalg.norm(tlx.convert_to_numpy(input), ord=p, axis=dim, keepdims=keepdim)

    op = tlx.convert_to_tensor(norm_np)
    if (tlx.BACKEND == "paddle"):
        op.stop_gradient = False
    else:
        op = tlx.Variable(op, 'op')
    return op


epsilon_F = 10 ** (-12)


def g(F):
    FF = tlx.ops.transpose(F) @ F
    FF_norm = norm(FF, p='fro')
    return (1 / (FF_norm + epsilon_F)) * FF
