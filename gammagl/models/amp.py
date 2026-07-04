# -*- encoding: utf-8 -*-
"""
Adaptive Message Passing (AMP)
------------------------------

Variational depth on graphs: a folded-normal mass is discretized on
:math:`\\{0,\\dots,L_{\\max}\\}` (renormalized). Message passing uses optional
edge-wise filters and GIN-style updates; graph readout uses concatenated global
add / max / mean pooling when ``global_aggregation`` is True.

This module follows GammaGL conventions: ``tensorlayerx`` layers and
``gammagl.layers.pool.glob`` for pooling.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import tensorlayerx as tlx
import tensorlayerx.nn as nn

from gammagl.layers.pool.glob import global_add_pool, global_max_pool, global_mean_pool


def _erf(x: tlx.Tensor) -> tlx.Tensor:
    if hasattr(tlx, "erf"):
        return tlx.erf(x)
    a1 = tlx.convert_to_tensor(0.254829592, dtype=tlx.float32)
    a2 = tlx.convert_to_tensor(-0.284496736, dtype=tlx.float32)
    a3 = tlx.convert_to_tensor(1.421413741, dtype=tlx.float32)
    a4 = tlx.convert_to_tensor(-1.453152027, dtype=tlx.float32)
    a5 = tlx.convert_to_tensor(1.061405429, dtype=tlx.float32)
    sign = tlx.sign(x)
    ax = tlx.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * tlx.exp(-ax * ax)
    return sign * y


def _softplus(x: tlx.Tensor) -> tlx.Tensor:
    return tlx.log(1.0 + tlx.exp(tlx.clip_by_value(x, -20.0, 20.0)))


def _folded_normal_cdf(value: tlx.Tensor, base_loc: tlx.Tensor, base_scale: tlx.Tensor) -> tlx.Tensor:
    b1 = 0.5 * (1.0 + _erf((value - base_loc) / (base_scale * math.sqrt(2.0) + 1e-12)))
    b2 = 0.5 * (1.0 + _erf((value + base_loc) / (base_scale * math.sqrt(2.0) + 1e-12)))
    return 0.5 * (2.0 * b1 - 1.0 + (2.0 * b2 - 1.0))


def _discretized_probs(
    max_depth: int,
    base_loc: tlx.Tensor,
    base_scale: tlx.Tensor,
    eps_mass: float = 1e-3,
) -> tlx.Tensor:
    k = tlx.arange(0, max_depth + 1, dtype=tlx.float32)
    k = tlx.reshape(k, (-1, 1))
    one = tlx.convert_to_tensor(1.0, dtype=tlx.float32)
    mass = _folded_normal_cdf(k + one, base_loc, base_scale) - _folded_normal_cdf(k, base_loc, base_scale)
    mass = mass + eps_mass
    mass = tlx.reshape(mass, (-1,))
    return mass / tlx.reduce_sum(mass)


class _EdgeFilterGINConv(nn.Module):
    """GIN-style layer with optional per-edge scalar gate (aligned with source nodes)."""

    def __init__(self, hidden_dim: int, train_eps: bool = True, name: Optional[str] = None):
        super().__init__(name=name)
        self.lin = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.train_eps = train_eps
        self.eps = tlx.nn.Parameter(tlx.zeros((1,), dtype=tlx.float32))

    def forward(
        self,
        x: tlx.Tensor,
        edge_index: tlx.Tensor,
        edge_msg_filter: Optional[tlx.Tensor],
    ) -> tlx.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        if edge_msg_filter is not None:
            if len(tlx.get_tensor_shape(edge_msg_filter)) == 1:
                edge_msg_filter = tlx.reshape(edge_msg_filter, (-1, 1))
            msg = edge_msg_filter * tlx.gather(x, src)
        else:
            msg = tlx.gather(x, src)
        n = tlx.get_tensor_shape(x)[0]
        aggr = tlx.unsorted_segment_sum(msg, dst, num_segments=n)
        ep = self.eps if self.train_eps else tlx.convert_to_tensor(0.0, dtype=tlx.float32)
        out = aggr + (1.0 + ep) * x
        out = self.lin(out)
        return tlx.tanh(out)


class _FilterBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.net = nn.Sequential(
            [
                nn.Linear(in_features=in_dim, out_features=hidden_dim),
                nn.Tanh(),
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            ]
        )

    def forward(self, x: tlx.Tensor) -> tlx.Tensor:
        return tlx.sigmoid(self.net(x))


def _log_prior_param_sum(module: nn.Module, theta_prior_scale: Optional[float]) -> tlx.Tensor:
    if theta_prior_scale is None:
        return tlx.convert_to_tensor(0.0, dtype=tlx.float32)
    s = float(theta_prior_scale)
    total = tlx.convert_to_tensor(0.0, dtype=tlx.float32)
    for w in module.trainable_weights:
        total = total - tlx.reduce_sum(tlx.square(w)) / (2.0 * s * s)
    return total


def amp_elbo_regression_loss(
    output_state: tlx.Tensor,
    targets: tlx.Tensor,
    log_p_theta_hidden: tlx.Tensor,
    log_p_theta_output: tlx.Tensor,
    log_p_L: tlx.Tensor,
    entropy_qL: tlx.Tensor,
    qL_probs: tlx.Tensor,
    n_obs: Union[tlx.Tensor, float],
) -> tlx.Tensor:
    r"""Negative ELBO for graph regression (Gaussian likelihood).

    ``output_state``: [num_graphs, num_layers, dim_target]
    ``qL_probs``: [1, num_layers]
    """
    two = tlx.convert_to_tensor(2.0, dtype=tlx.float32)
    if len(tlx.get_tensor_shape(targets)) == 1:
        targets = tlx.expand_dims(targets, -1)
    if len(tlx.get_tensor_shape(output_state)) == 2:
        output_state = tlx.expand_dims(output_state, -1)

    if isinstance(n_obs, (int, float)):
        n_obs_t = tlx.convert_to_tensor(float(n_obs), dtype=tlx.float32)
    else:
        n_obs_t = tlx.cast(n_obs, dtype=tlx.float32)

    n_layers = tlx.get_tensor_shape(output_state)[1]
    log_p_y_list = []
    for i in range(n_layers):
        diff = output_state[:, i, :] - targets
        se = tlx.reduce_sum(diff * diff, axis=1)
        log_p_y_list.append(-tlx.reduce_mean(se) / two * n_obs_t)
    log_p_y = tlx.stack(log_p_y_list, axis=0)
    log_p_y = tlx.reshape(log_p_y, (1, -1))

    elbo = log_p_y + log_p_theta_hidden + log_p_theta_output + log_p_L
    elbo = elbo * qL_probs
    elbo = tlx.reduce_sum(elbo, axis=1)
    elbo = elbo + entropy_qL
    elbo = elbo / n_obs_t
    return -tlx.reduce_mean(elbo)


class AMPModel(nn.Module):
    r"""Adaptive Message Passing for graph-level regression with variational depth.

    Parameters
    ----------
    in_channels: int
        Input node feature dimension.
    hidden_channels: int
        Hidden representation size.
    out_channels: int
        Graph-level output dimension.
    max_depth: int
        Maximum depth index :math:`L_{\max}`; outputs at :math:`L=0,\dots,L_{\max}`.
    theta_prior_scale: float, optional
        Gaussian weight prior std for ELBO aux terms; ``None`` disables priors.
    folded_loc_init: float
        Initial folded-normal location (non-negative).
    folded_scale_init: float
        Initial scale (positive via softplus).
    global_aggregation: bool
        If True, readout uses concatenated global add/max/mean pooling.
    filter_messages: str, optional
        ``embedding-no-weight-sharing`` or ``input_features``; ``None`` disables filters.
    name: str, optional
        Module name.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        max_depth: int = 8,
        theta_prior_scale: Optional[float] = 10.0,
        folded_loc_init: float = 5.0,
        folded_scale_init: float = 3.0,
        global_aggregation: bool = True,
        filter_messages: Optional[str] = "embedding-no-weight-sharing",
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)
        self.max_depth = int(max_depth)
        self.theta_prior_scale = theta_prior_scale
        self.global_aggregation = global_aggregation
        self.filter_messages = filter_messages

        self.base_loc = tlx.nn.Parameter(
            tlx.convert_to_tensor([[float(folded_loc_init)]], dtype=tlx.float32)
        )
        self._raw_scale = tlx.nn.Parameter(
            tlx.convert_to_tensor([[math.log(math.expm1(max(folded_scale_init - 0.5, 1e-6)))]], dtype=tlx.float32)
        )

        self.input_linear = nn.Linear(in_features=in_channels, out_features=hidden_channels)
        self.gin_layers = nn.ModuleList()
        for _ in range(max(0, self.max_depth - 1)):
            self.gin_layers.append(_EdgeFilterGINConv(hidden_channels, train_eps=True))

        def make_readout(in_dim: int) -> nn.Module:
            h = max(in_dim // 2, 1)
            return nn.Sequential(
                [
                    nn.Linear(in_features=in_dim, out_features=h),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=h, out_features=out_channels),
                    nn.LeakyReLU(),
                ]
            )

        readout_in_raw = (in_channels * 3) if global_aggregation else max(in_channels // 2, 1)
        readout_in_hid = (hidden_channels * 3) if global_aggregation else max(hidden_channels // 2, 1)
        self.readout_layers = nn.ModuleList()
        self.readout_layers.append(make_readout(readout_in_raw))
        for _ in range(self.max_depth):
            self.readout_layers.append(make_readout(readout_in_hid))

        if filter_messages:
            fdim = in_channels if filter_messages == "input_features" else hidden_channels
            self.filter_blocks = nn.ModuleList(
                [_FilterBlock(fdim, hidden_channels) for _ in range(self.max_depth)]
            )
        else:
            self.filter_blocks = None

    def base_scale(self) -> tlx.Tensor:
        const = tlx.convert_to_tensor(0.5, dtype=tlx.float32)
        return _softplus(self._raw_scale) + const

    def depth_probs(self) -> tlx.Tensor:
        return _discretized_probs(self.max_depth, self.base_loc, self.base_scale())

    def _readout(self, x: tlx.Tensor, batch: tlx.Tensor, layer_id: int) -> tlx.Tensor:
        if self.global_aggregation:
            h = tlx.concat(
                [
                    global_add_pool(x, batch),
                    global_max_pool(x, batch),
                    global_mean_pool(x, batch),
                ],
                axis=-1,
            )
        else:
            h = x
        return self.readout_layers[layer_id](h)

    def forward(
        self,
        x: tlx.Tensor,
        edge_index: tlx.Tensor,
        edge_attr: Optional[tlx.Tensor] = None,
        batch: Optional[tlx.Tensor] = None,
    ) -> tlx.Tensor:
        y, _, _ = self.forward_elbo(x, edge_index, edge_attr, batch)
        return y

    def forward_elbo(
        self,
        x: tlx.Tensor,
        edge_index: tlx.Tensor,
        edge_attr: Optional[tlx.Tensor] = None,
        batch: Optional[tlx.Tensor] = None,
    ) -> Tuple[tlx.Tensor, tlx.Tensor, Tuple[Any, ...]]:
        del edge_attr
        # OneHot / 部分后端在 GPU 上可能得到整型，Linear/matmul 需要浮点
        x = tlx.cast(x, tlx.float32)
        if batch is None:
            batch = tlx.zeros((tlx.get_tensor_shape(x)[0],), dtype=tlx.int64)

        q_probs = self.depth_probs()
        qL_b = tlx.expand_dims(q_probs, 0)

        q_sub = q_probs[1:]
        q_sub = q_sub / (tlx.reduce_sum(q_sub) + 1e-12)
        entropy_qL = -tlx.reduce_sum(q_sub * tlx.log(q_sub + 1e-12))
        entropy_qL = tlx.reshape(entropy_qL, (1,))

        first_state = x
        state = x
        out_list = []
        log_h_list = []
        log_o_list = []
        log_l_list = []
        log_theta_h_cum = tlx.convert_to_tensor(0.0, dtype=tlx.float32)
        log_theta_o_cum = tlx.convert_to_tensor(0.0, dtype=tlx.float32)

        for l in range(self.max_depth + 1):
            edge_msg_filter = None
            if self.filter_blocks is not None and l > 1:
                if self.filter_messages == "input_features":
                    mf = self.filter_blocks[l - 1](first_state)
                else:
                    mf = self.filter_blocks[l - 1](state)
                edge_msg_filter = tlx.gather(mf, edge_index[0])

            if l > 0:
                if l == 1:
                    state = self.input_linear(state)
                    lh = _log_prior_param_sum(self.input_linear, self.theta_prior_scale)
                else:
                    state = self.gin_layers[l - 2](state, edge_index, edge_msg_filter)
                    lh = _log_prior_param_sum(self.gin_layers[l - 2], self.theta_prior_scale)
                log_theta_h_cum = log_theta_h_cum + lh
            else:
                lh = tlx.convert_to_tensor(0.0, dtype=tlx.float32)

            lo = _log_prior_param_sum(self.readout_layers[l], self.theta_prior_scale)
            log_theta_o_cum = log_theta_o_cum + lo
            log_p_L = tlx.convert_to_tensor(0.0, dtype=tlx.float32)

            out_l = self._readout(state, batch, l)
            out_list.append(out_l)
            log_h_list.append(log_theta_h_cum)
            log_o_list.append(log_theta_o_cum)
            log_l_list.append(log_p_L)

        output_stack = tlx.stack(out_list, axis=1)
        log_h = tlx.reshape(tlx.stack(log_h_list, axis=0), (1, -1))
        log_o = tlx.reshape(tlx.stack(log_o_list, axis=0), (1, -1))
        log_l = tlx.reshape(tlx.stack(log_l_list, axis=0), (1, -1))

        y_expected = tlx.reduce_sum(output_stack * tlx.reshape(q_probs, (1, -1, 1)), axis=1)

        aux = (log_h, log_o, log_l, entropy_qL, qL_b)
        return y_expected, output_stack, aux
