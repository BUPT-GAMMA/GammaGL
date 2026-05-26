import numpy as np
import tensorlayerx as tlx
from defog_utils import (PlaceHolder, backend_one_hot, backend_nan_to_num, count_nonzero,
                         gather_last_dim)
from flow_utils import p_xt_g_x1, dt_p_xt_g_x1
from flow_matching_utils import sample_discrete_features


class RateMatrixDesigner:
    r"""Designs the rate matrix for CTMC-based sampling.

    Decomposes the rate matrix as ``R_t = R* + R^db + R^tg``:

    - ``R*``: Deterministic optimal flow rate.
    - ``R^db``: Detailed-balance stochastic rate (controlled by ``eta``).
    - ``R^tg``: Target guidance rate (controlled by ``omega``).

    Parameters
    ----------
    rdb : str
        Detailed-balance design type: ``'general'``, ``'marginal'``,
        ``'column'``, or ``'entry'``.
    rdb_crit : str
        Sub-criterion for ``rdb`` design (for ``column``: ``'max_marginal'``,
        ``'x_t'``, ``'abs_state'``, ``'p_x1_g_xt'``, ``'x_1'``,
        ``'p_xt_g_x1'``, ``'xhat_t'``; for ``entry``: ``'abs_state'``,
        ``'first'``).
    eta : float
        Stochasticity strength for ``R^db``.
    omega : float
        Target guidance strength for ``R^tg``.
    limit_dist : PlaceHolder
        Noise limit distribution.
    """
    def __init__(self, rdb='general', rdb_crit='max_marginal',
                 eta=0.0, omega=0.0, limit_dist=None):
        self.rdb = rdb
        self.rdb_crit = rdb_crit
        self.eta = eta
        self.omega = omega
        self.limit_dist = limit_dist
        self.num_classes_X = len(limit_dist.X) if limit_dist is not None else 0
        self.num_classes_E = len(limit_dist.E) if limit_dist is not None else 0

    def compute_graph_rate_matrix(self, t, node_mask, G_t, G_1_pred):
        r"""Compute the full rate matrix ``R_t = R* + R^db + R^tg``."""
        X_t, E_t = G_t
        X_1_pred, E_1_pred = G_1_pred

        # Get integer labels
        X_t_label = tlx.expand_dims(tlx.argmax(X_t, axis=-1), axis=-1)  # (bs, n, 1)
        E_t_label = tlx.expand_dims(tlx.argmax(E_t, axis=-1), axis=-1)  # (bs, n, n, 1)

        # Sample x_1 from predicted distributions
        sampled = sample_discrete_features(X_1_pred, E_1_pred, node_mask)
        X_1_sampled = sampled.X  # (bs, n) int
        E_1_sampled = sampled.E  # (bs, n, n) int

        # Compute shared variables
        dfm_vars = self._compute_dfm_variables(t, X_t_label, E_t_label,
                                                X_1_sampled, E_1_sampled)

        # R*
        Rstar_X, Rstar_E = self._compute_Rstar(dfm_vars)

        # R^db
        Rdb_X, Rdb_E = self._compute_RDB(X_t_label, E_t_label,
                                           X_1_pred, E_1_pred,
                                           X_1_sampled, E_1_sampled,
                                           node_mask, t, dfm_vars)

        # R^tg
        Rtg_X, Rtg_E = self._compute_R_tg(X_1_sampled, E_1_sampled,
                                            X_t_label, E_t_label, dfm_vars)

        R_t_X = Rstar_X + Rdb_X + Rtg_X
        R_t_E = Rstar_E + Rdb_E + Rtg_E

        R_t_X, R_t_E = self._stabilize(R_t_X, R_t_E, dfm_vars)

        return R_t_X, R_t_E

    def _compute_dfm_variables(self, t, X_t_label, E_t_label,
                                X_1_sampled, E_1_sampled):
        """Precompute shared quantities for rate matrix computation."""
        dX, dE = dt_p_xt_g_x1(X_1_sampled, E_1_sampled, self.limit_dist)

        dt_at_Xt = gather_last_dim(dX, X_t_label)
        dt_at_Et = gather_last_dim(dE, E_t_label)

        pX, pE = p_xt_g_x1(X_1_sampled, E_1_sampled, t, self.limit_dist)

        pt_at_Xt = gather_last_dim(pX, X_t_label)
        pt_at_Et = gather_last_dim(pE, E_t_label)

        Z_X = count_nonzero(pX, axis=-1)
        Z_E = count_nonzero(pE, axis=-1)

        return {
            'dt_p_vals_X': dX, 'dt_p_vals_E': dE,
            'dt_at_Xt': dt_at_Xt, 'dt_at_Et': dt_at_Et,
            'pt_vals_X': pX, 'pt_vals_E': pE,
            'pt_at_Xt': pt_at_Xt, 'pt_at_Et': pt_at_Et,
            'Z_X': Z_X, 'Z_E': Z_E,
        }

    def _compute_Rstar(self, dfm_vars):
        """Compute the deterministic optimal rate R*."""
        dX = dfm_vars['dt_p_vals_X']
        dE = dfm_vars['dt_p_vals_E']
        dt_at_Xt = dfm_vars['dt_at_Xt']
        dt_at_Et = dfm_vars['dt_at_Et']
        Z_X = dfm_vars['Z_X']
        Z_E = dfm_vars['Z_E']
        pt_at_Xt = dfm_vars['pt_at_Xt']
        pt_at_Et = dfm_vars['pt_at_Et']

        inner_X = dX - dt_at_Xt
        Rstar_numer_X = tlx.relu(inner_X)
        denom_X = tlx.expand_dims(Z_X, axis=-1) * pt_at_Xt
        denom_X = tlx.where(denom_X == 0, tlx.ones_like(denom_X), denom_X)
        Rstar_X = Rstar_numer_X / denom_X

        inner_E = dE - dt_at_Et
        Rstar_numer_E = tlx.relu(inner_E)
        denom_E = tlx.expand_dims(Z_E, axis=-1) * pt_at_Et
        denom_E = tlx.where(denom_E == 0, tlx.ones_like(denom_E), denom_E)
        Rstar_E = Rstar_numer_E / denom_E

        return Rstar_X, Rstar_E

    def _compute_RDB(self, X_t_label, E_t_label, X_1_pred, E_1_pred,
                     X_1_sampled, E_1_sampled, node_mask, t, dfm_vars):
        """Compute the detailed-balance stochastic rate R^db.

        Supports ``'general'``, ``'marginal'``, ``'column'`` (with 7 sub-criteria),
        and ``'entry'`` (with ``'abs_state'`` and ``'first'``) designs.
        """
        if self.eta == 0:
            zeros_X = tlx.zeros_like(dfm_vars['pt_vals_X'])
            zeros_E = tlx.zeros_like(dfm_vars['pt_vals_E'])
            return zeros_X, zeros_E

        pX = dfm_vars['pt_vals_X']  # (bs, n, dx)
        pE = dfm_vars['pt_vals_E']  # (bs, n, n, de)

        dx = pX.shape[-1]
        de = pE.shape[-1]

        if self.rdb == 'general':
            x_mask = tlx.ones_like(pX)
            e_mask = tlx.ones_like(pE)

        elif self.rdb == 'marginal':
            limit_X = tlx.reshape(self.limit_dist.X, [1, 1, -1])
            limit_E = tlx.reshape(self.limit_dist.E, [1, 1, 1, -1])
            limit_at_Xt = gather_last_dim(
                tlx.tile(limit_X, [pX.shape[0], pX.shape[1], 1]),
                X_t_label
            )
            limit_at_Et = gather_last_dim(
                tlx.tile(limit_E, [pE.shape[0], pE.shape[1], pE.shape[2], 1]),
                E_t_label
            )
            x_mask = tlx.cast(limit_X > limit_at_Xt, tlx.float32)
            e_mask = tlx.cast(limit_E > limit_at_Et, tlx.float32)

        elif self.rdb == 'column':
            # Determine column indices based on sub-criterion
            if self.rdb_crit == 'max_marginal':
                limit_X_np = tlx.convert_to_numpy(self.limit_dist.X)
                limit_E_np = tlx.convert_to_numpy(self.limit_dist.E)
                x_col = np.full_like(tlx.convert_to_numpy(X_t_label).squeeze(-1),
                                     limit_X_np.argmax(), dtype=np.int64)
                e_col = np.full_like(tlx.convert_to_numpy(E_t_label).squeeze(-1),
                                     limit_E_np.argmax(), dtype=np.int64)
                x_column_idxs = tlx.convert_to_tensor(x_col[..., np.newaxis])
                e_column_idxs = tlx.convert_to_tensor(e_col[..., np.newaxis])

            elif self.rdb_crit == 'x_t':
                x_column_idxs = X_t_label
                e_column_idxs = E_t_label

            elif self.rdb_crit == 'abs_state':
                x_column_idxs = tlx.ones_like(X_t_label) * (dx - 1)
                e_column_idxs = tlx.ones_like(E_t_label) * (de - 1)

            elif self.rdb_crit == 'p_x1_g_xt':
                x_column_idxs = tlx.expand_dims(tlx.argmax(X_1_pred, axis=-1), axis=-1)
                e_column_idxs = tlx.expand_dims(tlx.argmax(E_1_pred, axis=-1), axis=-1)

            elif self.rdb_crit == 'x_1':
                x_column_idxs = tlx.expand_dims(X_1_sampled, axis=-1)
                e_column_idxs = tlx.expand_dims(E_1_sampled, axis=-1)

            elif self.rdb_crit == 'p_xt_g_x1':
                x_column_idxs = tlx.expand_dims(tlx.argmax(pX, axis=-1), axis=-1)
                e_column_idxs = tlx.expand_dims(tlx.argmax(pE, axis=-1), axis=-1)

            elif self.rdb_crit == 'xhat_t':
                sampled_hat = sample_discrete_features(pX, pE, node_mask)
                x_column_idxs = tlx.expand_dims(sampled_hat.X, axis=-1)
                e_column_idxs = tlx.expand_dims(sampled_hat.E, axis=-1)

            else:
                raise NotImplementedError(f"rdb_crit '{self.rdb_crit}' not implemented for column")

            # Create mask: one-hot at column_idx, plus keep current state
            x_col_squeezed = tlx.reshape(x_column_idxs, [pX.shape[0], pX.shape[1]])
            e_col_squeezed = tlx.reshape(e_column_idxs, pE.shape[:3])

            x_mask = backend_one_hot(x_col_squeezed, dx)
            e_mask = backend_one_hot(e_col_squeezed, de)

            # Also keep current state
            eq_x = tlx.cast(tlx.reshape(x_column_idxs == X_t_label, x_mask.shape[:-1] + (1,)),
                            tlx.float32)
            eq_e = tlx.cast(tlx.reshape(e_column_idxs == E_t_label, e_mask.shape[:-1] + (1,)),
                            tlx.float32)
            x_mask = tlx.maximum(x_mask, eq_x)
            e_mask = tlx.maximum(e_mask, eq_e)

        elif self.rdb == 'entry':
            if self.rdb_crit == 'abs_state':
                x_masked_idx = dx - 1
                e_masked_idx = de - 1
                x1_idxs = tlx.expand_dims(X_1_sampled, axis=-1)  # (bs, n, 1)
                e1_idxs = tlx.expand_dims(E_1_sampled, axis=-1)  # (bs, n, n, 1)
            elif self.rdb_crit == 'first':
                x_masked_idx = 0
                e_masked_idx = 0
                x1_idxs = tlx.expand_dims(X_1_sampled, axis=-1)
                e1_idxs = tlx.expand_dims(E_1_sampled, axis=-1)
            else:
                raise NotImplementedError(f"rdb_crit '{self.rdb_crit}' not implemented for entry")

            # Build mask via numpy for advanced indexing
            x_mask_np = np.zeros_like(tlx.convert_to_numpy(pX))
            e_mask_np = np.zeros_like(tlx.convert_to_numpy(pE))

            X_t_np = tlx.convert_to_numpy(X_t_label).squeeze(-1).astype(np.int64)
            E_t_np = tlx.convert_to_numpy(E_t_label).squeeze(-1).astype(np.int64)
            x1_np = tlx.convert_to_numpy(x1_idxs).squeeze(-1).astype(np.int64)
            e1_np = tlx.convert_to_numpy(e1_idxs).squeeze(-1).astype(np.int64)

            bs = x_mask_np.shape[0]

            # X: swap between current and target
            for b in range(bs):
                n = x_mask_np.shape[1]
                for i in range(n):
                    if X_t_np[b, i] == x1_np[b, i]:
                        # Current == target: mark the absorbing state
                        if x_masked_idx < dx:
                            x_mask_np[b, i, x_masked_idx] = 1.0
                    elif X_t_np[b, i] == x_masked_idx:
                        # Current is the absorbing state: mark target
                        x_mask_np[b, i, x1_np[b, i]] = 1.0

            for b in range(bs):
                ni, nj = e_mask_np.shape[1], e_mask_np.shape[2]
                for i in range(ni):
                    for j in range(nj):
                        if E_t_np[b, i, j] == e1_np[b, i, j]:
                            if e_masked_idx < de:
                                e_mask_np[b, i, j, e_masked_idx] = 1.0
                        elif E_t_np[b, i, j] == e_masked_idx:
                            e_mask_np[b, i, j, e1_np[b, i, j]] = 1.0

            x_mask = tlx.convert_to_tensor(x_mask_np)
            e_mask = tlx.convert_to_tensor(e_mask_np)

        else:
            raise NotImplementedError(f"rdb type '{self.rdb}' not implemented")

        Rdb_X = pX * x_mask * self.eta
        Rdb_E = pE * e_mask * self.eta

        return Rdb_X, Rdb_E

    def _compute_R_tg(self, X_1_sampled, E_1_sampled, X_t_label, E_t_label,
                      dfm_vars):
        """Compute the target guidance rate R^tg."""
        if self.omega == 0:
            zeros_X = tlx.zeros_like(dfm_vars['pt_vals_X'])
            zeros_E = tlx.zeros_like(dfm_vars['pt_vals_E'])
            return zeros_X, zeros_E

        dx = self.num_classes_X
        de = self.num_classes_E

        X1_oh = backend_one_hot(X_1_sampled, dx)
        E1_oh = backend_one_hot(E_1_sampled, de)

        X1_exp = tlx.expand_dims(X_1_sampled, axis=-1)
        mask_X = tlx.cast(X1_exp != X_t_label, tlx.float32)
        E1_exp = tlx.expand_dims(E_1_sampled, axis=-1)
        mask_E = tlx.cast(E1_exp != E_t_label, tlx.float32)

        numer_X = X1_oh * self.omega * mask_X
        denom_X = tlx.expand_dims(dfm_vars['Z_X'], axis=-1) * dfm_vars['pt_at_Xt']
        denom_X = tlx.where(denom_X == 0, tlx.ones_like(denom_X), denom_X)
        Rtg_X = numer_X / denom_X

        numer_E = E1_oh * self.omega * mask_E
        denom_E = tlx.expand_dims(dfm_vars['Z_E'], axis=-1) * dfm_vars['pt_at_Et']
        denom_E = tlx.where(denom_E == 0, tlx.ones_like(denom_E), denom_E)
        Rtg_E = numer_E / denom_E

        return Rtg_X, Rtg_E

    def _stabilize(self, R_X, R_E, dfm_vars):
        """Post-processing to avoid numerical instabilities."""
        R_X = backend_nan_to_num(R_X)
        R_E = backend_nan_to_num(R_E)

        R_X = tlx.where(R_X > 1e5, tlx.zeros_like(R_X), R_X)
        R_E = tlx.where(R_E > 1e5, tlx.zeros_like(R_E), R_E)

        pt_at_Xt = dfm_vars['pt_at_Xt']
        zero_mask_X = tlx.cast(pt_at_Xt == 0, tlx.float32)
        R_X = R_X * (1.0 - zero_mask_X)

        pt_at_Et = dfm_vars['pt_at_Et']
        zero_mask_E = tlx.cast(pt_at_Et == 0, tlx.float32)
        R_E = R_E * (1.0 - zero_mask_E)

        pX = dfm_vars['pt_vals_X']
        pE = dfm_vars['pt_vals_E']
        col_mask_X = tlx.cast(pX == 0, tlx.float32)
        R_X = R_X * (1.0 - col_mask_X)
        col_mask_E = tlx.cast(pE == 0, tlx.float32)
        R_E = R_E * (1.0 - col_mask_E)

        return R_X, R_E
