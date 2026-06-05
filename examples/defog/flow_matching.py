from defog_utils import PlaceHolder, apply_node_mask
import torch
import math
import numpy as np
import tensorlayerx as tlx




def p_xt_g_x1(X1, E1, t, limit_dist):
    r"""Compute the conditional probability ``p(x_t | x_1)`` under linear flow.

    Linear interpolation: ``p(x_t | x_1) = t * onehot(x_1) + (1-t) * limit_dist``.

    Parameters
    ----------
    X1 : tensor
        Integer node labels, shape ``(bs, n)``.
    E1 : tensor
        Integer edge labels, shape ``(bs, n, n)``.
    t : tensor
        Time values in [0,1], shape ``(bs, 1)``.
    limit_dist : PlaceHolder
        Noise distribution with ``.X`` of shape ``(dx,)`` and ``.E`` of shape ``(de,)``.

    Returns
    -------
    tuple
        ``(prob_X, prob_E)`` with shapes ``(bs, n, dx)`` and ``(bs, n, n, de)``.
    """
    dx = len(limit_dist.X)
    de = len(limit_dist.E)

    # t_time shape: (bs, 1, 1) for broadcasting with X
    t_squeezed = torch.reshape(t, [-1])  # (bs,)
    t_x = torch.reshape(t_squeezed, [-1, 1, 1])       # (bs, 1, 1)
    t_e = torch.reshape(t_squeezed, [-1, 1, 1, 1])    # (bs, 1, 1, 1)

    X1_onehot = torch.nn.functional.one_hot(X1, num_classes=dx).float()  # (bs, n, dx)
    E1_onehot = torch.nn.functional.one_hot(E1, num_classes=de).float()  # (bs, n, n, de)

    limit_X = torch.reshape(limit_dist.X.to(torch.float32), [1, 1, dx])    # (1, 1, dx)
    limit_E = torch.reshape(limit_dist.E.to(torch.float32), [1, 1, 1, de]) # (1, 1, 1, de)

    prob_X = t_x * X1_onehot + (1.0 - t_x) * limit_X
    prob_E = t_e * E1_onehot + (1.0 - t_e) * limit_E

    prob_X = torch.clamp(prob_X, min=0.0, max=1.0)
    prob_E = torch.clamp(prob_E, min=0.0, max=1.0)

    return prob_X, prob_E


def dt_p_xt_g_x1(X1, E1, limit_dist):
    r"""Compute the time derivative ``d/dt p(x_t | x_1)``.

    Since the interpolation is linear, the derivative is constant:
    ``d/dt p(x_t | x_1) = onehot(x_1) - limit_dist``.

    Parameters
    ----------
    X1 : tensor
        Integer node labels, shape ``(bs, n)``.
    E1 : tensor
        Integer edge labels, shape ``(bs, n, n)``.
    limit_dist : PlaceHolder
        Noise distribution.

    Returns
    -------
    tuple
        ``(dX, dE)`` with shapes ``(bs, n, dx)`` and ``(bs, n, n, de)``.
    """
    dx = len(limit_dist.X)
    de = len(limit_dist.E)

    X1_onehot = torch.nn.functional.one_hot(X1, num_classes=dx).float()  # (bs, n, dx)
    E1_onehot = torch.nn.functional.one_hot(E1, num_classes=de).float()  # (bs, n, n, de)

    limit_X = torch.reshape(limit_dist.X.to(torch.float32), [1, 1, dx])
    limit_E = torch.reshape(limit_dist.E.to(torch.float32), [1, 1, 1, de])

    dX = X1_onehot - limit_X
    dE = E1_onehot - limit_E

    return dX, dE




def assert_correctly_masked(variable, node_mask):
    r"""Debug assertion: masked positions should be near zero.

    Parameters
    ----------
    variable : tensor
        Tensor to check.
    node_mask : tensor
        Boolean mask ``(bs, n)``.
    """
    x_mask = torch.unsqueeze(node_mask.to(torch.float32), dim=-1)
    inv_mask = 1.0 - x_mask
    masked_vals = variable * inv_mask
    max_val = float(torch.max(torch.abs(masked_vals)))
    assert max_val < 1e-4, f"Masked values not zero: max={max_val}"


def sample_discrete_feature_noise(limit_dist, node_mask):
    r"""Sample noise from the limit distribution.

    Parameters
    ----------
    limit_dist : PlaceHolder
        Noise distribution with ``.X`` shape ``(dx,)``, ``.E`` shape ``(de,)``.
    node_mask : tensor
        Boolean mask ``(bs, n)``.

    Returns
    -------
    PlaceHolder
        Sampled one-hot noise ``(X, E, y)`` masked.
    """
    nm = node_mask.to(torch.float32)
    bs = nm.shape[0]
    n = nm.shape[1]

    dx = len(limit_dist.X)
    de = len(limit_dist.E)

    # Sample nodes
    x_probs = limit_dist.X.to(torch.float32)
    x_probs = torch.tile(torch.reshape(x_probs, [1, dx]), [bs * n, 1])  # (bs*n, dx)
    U_X = torch.multinomial(x_probs, 1)  # (bs*n, 1)
    U_X = torch.reshape(U_X, [bs, n])  # (bs, n)

    # Sample edges
    e_probs = limit_dist.E.to(torch.float32)
    e_probs = torch.tile(torch.reshape(e_probs, [1, de]), [bs * n * n, 1])
    U_E = torch.multinomial(e_probs, 1)
    U_E = torch.reshape(U_E, [bs, n, n])

    # Convert to one-hot
    U_X_oh = torch.nn.functional.one_hot(U_X, num_classes=dx).float()  # (bs, n, dx)
    U_E_oh = torch.nn.functional.one_hot(U_E, num_classes=de).float()  # (bs, n, n, de)

    # Symmetrize E: keep upper triangle, add transpose
    U_E_oh = torch.triu(U_E_oh, diagonal=1)
    U_E_oh = U_E_oh + U_E_oh.permute(*[0, 2, 1, 3])

    U_y = torch.zeros([bs, 0], dtype=torch.float32)

    return PlaceHolder(X=U_X_oh, E=U_E_oh, y=U_y).mask(node_mask)


def sample_discrete_features(probX, probE, node_mask, mask=False):
    r"""Sample discrete features (integer class labels) from probability distributions.

    Parameters
    ----------
    probX : tensor
        Node class probabilities ``(bs, n, dx)``.
    probE : tensor
        Edge class probabilities ``(bs, n, n, de)``.
    node_mask : tensor
        Boolean mask ``(bs, n)``.
    mask : bool
        If True, zero out masked positions after sampling.

    Returns
    -------
    PlaceHolder
        Sampled integer indices ``X: (bs, n)``, ``E: (bs, n, n)``.
    """
    bs = probX.shape[0]
    n = probX.shape[1]
    dx = probX.shape[2]
    de = probE.shape[3]

    # Handle masked nodes: set uniform probability
    nm = node_mask.to(torch.float32)
    nm_exp = torch.unsqueeze(nm, dim=-1)  # (bs, n, 1)
    uniform_x = torch.ones_like(probX) / dx
    probX = probX * nm_exp + uniform_x * (1.0 - nm_exp)

    # Sample nodes
    probX_flat = torch.reshape(probX, [bs * n, dx])
    X_t = torch.multinomial(probX_flat, 1)  # (bs*n, 1)
    X_t = torch.reshape(X_t, [bs, n])

    # Handle masked edges: set uniform probability for invalid pairs and diagonal
    inv_edge_mask = 1.0 - torch.unsqueeze(nm, dim=2) * torch.unsqueeze(nm, dim=1)
    inv_edge_mask = torch.unsqueeze(inv_edge_mask, dim=-1)  # (bs, n, n, 1)

    eye_n = torch.eye(n)
    diag_mask = torch.reshape(eye_n, [1, n, n, 1])
    diag_mask = torch.tile(diag_mask, [bs, 1, 1, 1])

    uniform_e = torch.ones_like(probE) / de
    probE = probE * (1.0 - inv_edge_mask) + uniform_e * inv_edge_mask
    probE = probE * (1.0 - diag_mask) + uniform_e * diag_mask

    # Sample edges
    probE_flat = torch.reshape(probE, [bs * n * n, de])
    E_t = torch.multinomial(probE_flat, 1)
    E_t = torch.reshape(E_t, [bs, n, n])

    # Symmetrize: keep upper triangle, mirror to lower
    E_t_int = E_t.to(torch.int64)
    idx = torch.arange(n, device=E_t.device)
    mask_diag = (idx.unsqueeze(0).unsqueeze(-1) == idx.unsqueeze(0).unsqueeze(1))
    E_t_int[mask_diag] = 0
    E_t_int = torch.triu(E_t_int, diagonal=1)

    E_t_int = E_t_int + E_t_int.permute(*[0, 2, 1])

    if mask:
        nm_int = node_mask.to(torch.int64)
        X_t = X_t.to(torch.int64) * nm_int
        em = torch.unsqueeze(nm_int, dim=2) * torch.unsqueeze(nm_int, dim=1)
        E_t_int = E_t_int * em

    X_t = X_t.to(torch.int64)
    E_t_int = E_t_int.to(torch.int64)
    U_y = torch.zeros([bs, 0], dtype=torch.float32)

    return PlaceHolder(X=X_t, E=E_t_int, y=U_y)




class TimeDistorter:
    r"""Time distortion for training and sampling schedules.

    Transforms uniform time ``t in [0, 1]`` to reshape the
    training/sampling schedule.

    Parameters
    ----------
    train_distortion : str
        Distortion type for training. One of ``'identity'``, ``'cos'``,
        ``'revcos'``, ``'polyinc'``, ``'polydec'``.
    sample_distortion : str
        Distortion type for sampling.
    """
    def __init__(self, train_distortion='identity', sample_distortion='identity'):
        self.train_distortion = train_distortion
        self.sample_distortion = sample_distortion

    def train_ft(self, batch_size):
        r"""Sample distorted time values for training.

        Parameters
        ----------
        batch_size : int
            Number of time samples to generate.

        Returns
        -------
        tensor
            Distorted time values of shape ``(batch_size, 1)``.
        """
        t_uniform = torch.tensor(
            np.random.uniform(0, 1, size=(batch_size, 1)).astype(np.float32)
        )
        return self.apply_distortion(t_uniform, self.train_distortion)

    def sample_ft(self, t, sample_distortion=None):
        r"""Apply distortion to a time value during sampling.

        Parameters
        ----------
        t : tensor
            Time values.
        sample_distortion : str, optional
            Override distortion type. Defaults to ``self.sample_distortion``.

        Returns
        -------
        tensor
            Distorted time values.
        """
        if sample_distortion is None:
            sample_distortion = self.sample_distortion
        return self.apply_distortion(t, sample_distortion)

    def apply_distortion(self, t, distortion_type):
        r"""Apply a time distortion function.

        Parameters
        ----------
        t : tensor
            Time values in [0, 1].
        distortion_type : str
            One of ``'identity'``, ``'cos'``, ``'revcos'``, ``'polyinc'``, ``'polydec'``.

        Returns
        -------
        tensor
            Distorted time values in [0, 1].
        """
        if distortion_type == 'identity':
            return t
        elif distortion_type == 'cos':
            return (1.0 - torch.cos(t * math.pi)) / 2.0
        elif distortion_type == 'revcos':
            return 2.0 * t - (1.0 - torch.cos(t * math.pi)) / 2.0
        elif distortion_type == 'polyinc':
            return t ** 2
        elif distortion_type == 'polydec':
            return 2.0 * t - t ** 2
        else:
            raise ValueError(f"Unknown distortion type: {distortion_type}")




class NoiseDistribution:
    r"""Noise/limit distribution for the discrete flow matching process.

    Supports multiple transition types that define the noise distribution at ``t=0``.

    Parameters
    ----------
    model_transition : str
        Transition type. One of ``'uniform'``, ``'marginal'``, ``'absorbing'``,
        ``'absorbfirst'``, ``'argmax'``, ``'edge_marginal'``, ``'node_marginal'``.
    dataset_infos : object
        Dataset info object with attributes ``output_dims``, ``node_types``,
        ``edge_types``.
    """
    def __init__(self, model_transition, dataset_infos):
        self.transition = model_transition

        output_dims = dataset_infos['output_dims']
        x_num_classes = output_dims['X']
        e_num_classes = output_dims['E']
        y_num_classes = output_dims.get('y', 0)

        self.x_num_classes = x_num_classes
        self.e_num_classes = e_num_classes
        self.y_num_classes = y_num_classes

        self.x_added_classes = 0
        self.e_added_classes = 0
        self.y_added_classes = 0

        node_types = dataset_infos.get('node_types', None)
        edge_types = dataset_infos.get('edge_types', None)

        if model_transition == 'uniform':
            x_limit = np.ones(x_num_classes, dtype=np.float32) / x_num_classes
            e_limit = np.ones(e_num_classes, dtype=np.float32) / e_num_classes

        elif model_transition == 'absorbfirst':
            x_limit = np.zeros(x_num_classes, dtype=np.float32)
            x_limit[0] = 1.0
            e_limit = np.zeros(e_num_classes, dtype=np.float32)
            e_limit[0] = 1.0

        elif model_transition == 'argmax':
            node_marginal = node_types / node_types.sum()
            edge_marginal = edge_types / edge_types.sum()
            x_limit = np.zeros(x_num_classes, dtype=np.float32)
            x_limit[np.argmax(node_marginal)] = 1.0
            e_limit = np.zeros(e_num_classes, dtype=np.float32)
            e_limit[np.argmax(edge_marginal)] = 1.0

        elif model_transition == 'absorbing':
            if x_num_classes > 1:
                self.x_added_classes = 1
                self.x_num_classes = x_num_classes + 1
            if e_num_classes > 1:
                self.e_added_classes = 1
                self.e_num_classes = e_num_classes + 1
            x_limit = np.zeros(self.x_num_classes, dtype=np.float32)
            x_limit[-1] = 1.0
            e_limit = np.zeros(self.e_num_classes, dtype=np.float32)
            e_limit[-1] = 1.0

        elif model_transition == 'marginal':
            x_limit = (node_types / node_types.sum()).astype(np.float32)
            e_limit = (edge_types / edge_types.sum()).astype(np.float32)

        elif model_transition == 'edge_marginal':
            x_limit = np.ones(x_num_classes, dtype=np.float32) / x_num_classes
            e_limit = (edge_types / edge_types.sum()).astype(np.float32)

        elif model_transition == 'node_marginal':
            x_limit = (node_types / node_types.sum()).astype(np.float32)
            e_limit = np.ones(e_num_classes, dtype=np.float32) / e_num_classes

        else:
            raise ValueError(f"Unknown transition type: {model_transition}")

        if y_num_classes > 0:
            y_limit = np.ones(y_num_classes, dtype=np.float32) / y_num_classes
        else:
            y_limit = np.zeros(0, dtype=np.float32)

        self.limit_dist = PlaceHolder(
            X=torch.tensor(x_limit),
            E=torch.tensor(e_limit),
            y=torch.tensor(y_limit),
        )

        print(f"[NoiseDistribution] transition={model_transition}")
        print(f"  X limit: {x_limit}")
        print(f"  E limit: {e_limit}")

    def update_dataset_infos(self, dataset_infos):
        r"""Update dataset_infos to account for virtual absorbing classes.

        When using the absorbing transition, the atom decoder is extended with
        a virtual token so downstream molecular tooling sees the correct node
        vocabulary.

        Parameters
        ----------
        dataset_infos : dict
            Dataset info dict (modified in-place).
        """
        if self.transition != 'absorbing':
            return

        if dataset_infos.get('atom_decoder', None) is not None and self.x_added_classes > 0:
            dataset_infos['atom_decoder'] = list(dataset_infos['atom_decoder']) + ['Y'] * self.x_added_classes

    def update_input_output_dims(self, input_dims):
        r"""Update input dims to account for added virtual classes."""
        input_dims['X'] = input_dims['X'] + self.x_added_classes
        input_dims['E'] = input_dims['E'] + self.e_added_classes
        input_dims['y'] = input_dims['y'] + self.y_added_classes

    def get_limit_dist(self):
        r"""Return the limit distribution."""
        return self.limit_dist

    def get_noise_dims(self):
        r"""Return the noise distribution dimensions."""
        return {
            'X': len(self.limit_dist.X),
            'E': len(self.limit_dist.E),
            'y': len(self.limit_dist.y),
        }

    def ignore_virtual_classes(self, X, E, y=None):
        r"""Remove virtual absorbing-state classes from X and E."""
        if self.transition != 'absorbing':
            return (X, E, y) if y is not None else (X, E)

        if self.x_added_classes > 0:
            X = X[..., :-self.x_added_classes]
        if self.e_added_classes > 0:
            E = E[..., :-self.e_added_classes]
        if y is not None:
            if self.y_added_classes > 0:
                y = y[..., :-self.y_added_classes]
            return X, E, y
        return X, E

    def add_virtual_classes(self, X, E, y=None):
        r"""Add virtual absorbing-state classes to X and E."""
        if self.transition != 'absorbing':
            return (X, E, y) if y is not None else (X, E)

        if self.x_added_classes > 0:
            zeros_x = torch.zeros(list(X.shape[:-1]) + [self.x_added_classes],
                                dtype=X.dtype)
            X = torch.cat([X, zeros_x], dim=-1)
        if self.e_added_classes > 0:
            zeros_e = torch.zeros(list(E.shape[:-1]) + [self.e_added_classes],
                                dtype=E.dtype)
            E = torch.cat([E, zeros_e], dim=-1)
        if y is not None:
            return X, E, y
        return X, E


def apply_noise(X, E, y, node_mask, limit_dist, time_distorter):
    r"""Apply noise to clean graph data for training.

    Parameters
    ----------
    X : tensor
        Clean node features ``(bs, n, dx)`` (one-hot).
    E : tensor
        Clean edge features ``(bs, n, n, de)`` (one-hot).
    y : tensor
        Global features ``(bs, dy)``.
    node_mask : tensor
        Boolean mask ``(bs, n)``.
    limit_dist : PlaceHolder
        Noise limit distribution.
    time_distorter : TimeDistorter
        Time distortion function.

    Returns
    -------
    dict
        Noisy data with keys ``'t'``, ``'X_t'``, ``'E_t'``, ``'y_t'``, ``'node_mask'``.
    """
    bs = X.shape[0]

    # Move limit_dist to the same device as input (needed for DataParallel)
    device = X.device if hasattr(X, 'device') else None
    if device is not None and hasattr(limit_dist, 'X') and hasattr(limit_dist.X, 'to'):
        if limit_dist.X.device != device:
            limit_dist = PlaceHolder(X=limit_dist.X.to(device), E=limit_dist.E.to(device), y=limit_dist.y)

    # Sample time
    t = time_distorter.train_ft(bs)  # (bs, 1)

    # Get clean integer labels
    X_1 = torch.argmax(X, dim=-1)   # (bs, n)
    E_1 = torch.argmax(E, dim=-1)   # (bs, n, n)

    # Compute transition probabilities
    prob_X, prob_E = p_xt_g_x1(X_1, E_1, t, limit_dist)
    # prob_X: (bs, n, dx), prob_E: (bs, n, n, de)

    # Sample noisy features
    sampled = sample_discrete_features(prob_X, prob_E, node_mask)
    X_t_int = sampled.X  # (bs, n)
    E_t_int = sampled.E  # (bs, n, n)

    dx = len(limit_dist.X)
    de = len(limit_dist.E)

    # One-hot encode
    X_t = torch.nn.functional.one_hot(X_t_int, num_classes=dx).float()  # (bs, n, dx)
    E_t = torch.nn.functional.one_hot(E_t_int, num_classes=de).float()  # (bs, n, n, de)

    # Mask
    X_t, E_t = apply_node_mask(X_t, E_t, node_mask)

    y_t = y if y is not None else torch.zeros([bs, 0], dtype=torch.float32)

    return {
        't': t,
        'X_t': X_t,
        'E_t': E_t,
        'y_t': y_t,
        'node_mask': node_mask,
    }





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
        ``'first'``.
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
        X_t_label = torch.unsqueeze(torch.argmax(X_t, dim=-1), dim=-1)  # (bs, n, 1)
        E_t_label = torch.unsqueeze(torch.argmax(E_t, dim=-1), dim=-1)  # (bs, n, n, 1)

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

        dt_at_Xt = torch.gather(dX, -1, X_t_label)
        dt_at_Et = torch.gather(dE, -1, E_t_label)

        pX, pE = p_xt_g_x1(X_1_sampled, E_1_sampled, t, self.limit_dist)

        pt_at_Xt = torch.gather(pX, -1, X_t_label)
        pt_at_Et = torch.gather(pE, -1, E_t_label)

        Z_X = (pX != 0).sum(dim=-1)
        Z_E = (pE != 0).sum(dim=-1)

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
        Rstar_numer_X = torch.relu(inner_X)
        denom_X = torch.unsqueeze(Z_X, dim=-1) * pt_at_Xt
        denom_X = torch.where(denom_X == 0, torch.ones_like(denom_X), denom_X)
        Rstar_X = Rstar_numer_X / denom_X

        inner_E = dE - dt_at_Et
        Rstar_numer_E = torch.relu(inner_E)
        denom_E = torch.unsqueeze(Z_E, dim=-1) * pt_at_Et
        denom_E = torch.where(denom_E == 0, torch.ones_like(denom_E), denom_E)
        Rstar_E = Rstar_numer_E / denom_E

        return Rstar_X, Rstar_E

    def _compute_RDB(self, X_t_label, E_t_label, X_1_pred, E_1_pred,
                     X_1_sampled, E_1_sampled, node_mask, t, dfm_vars):
        """Compute the detailed-balance stochastic rate R^db.

        Supports ``'general'``, ``'marginal'``, ``'column'`` (with 7 sub-criteria),
        and ``'entry'`` (with ``'abs_state'`` and ``'first'``) designs.
        """
        if self.eta == 0:
            zeros_X = torch.zeros_like(dfm_vars['pt_vals_X'])
            zeros_E = torch.zeros_like(dfm_vars['pt_vals_E'])
            return zeros_X, zeros_E

        pX = dfm_vars['pt_vals_X']  # (bs, n, dx)
        pE = dfm_vars['pt_vals_E']  # (bs, n, n, de)

        dx = pX.shape[-1]
        de = pE.shape[-1]

        if self.rdb == 'general':
            x_mask = torch.ones_like(pX)
            e_mask = torch.ones_like(pE)

        elif self.rdb == 'marginal':
            limit_X = torch.reshape(self.limit_dist.X, [1, 1, -1])
            limit_E = torch.reshape(self.limit_dist.E, [1, 1, 1, -1])
            limit_at_Xt = torch.gather(
                torch.tile(limit_X, [pX.shape[0], pX.shape[1], 1]), -1, X_t_label
            )
            limit_at_Et = torch.gather(
                torch.tile(limit_E, [pE.shape[0], pE.shape[1], pE.shape[2], 1]), -1, E_t_label
            )
            x_mask = (limit_X > limit_at_Xt).to(torch.float32)
            e_mask = (limit_E > limit_at_Et).to(torch.float32)

        elif self.rdb == 'column':
            # Determine column indices based on sub-criterion
            if self.rdb_crit == 'max_marginal':
                limit_X_np = self.limit_dist.X.detach().cpu().numpy()
                limit_E_np = self.limit_dist.E.detach().cpu().numpy()
                x_col = np.full_like(X_t_label.detach().cpu().numpy().squeeze(-1),
                                     limit_X_np.argmax(), dtype=np.int64)
                e_col = np.full_like(E_t_label.detach().cpu().numpy().squeeze(-1),
                                     limit_E_np.argmax(), dtype=np.int64)
                x_column_idxs = torch.tensor(x_col[..., np.newaxis])
                e_column_idxs = torch.tensor(e_col[..., np.newaxis])

            elif self.rdb_crit == 'x_t':
                x_column_idxs = X_t_label
                e_column_idxs = E_t_label

            elif self.rdb_crit == 'abs_state':
                x_column_idxs = torch.ones_like(X_t_label) * (dx - 1)
                e_column_idxs = torch.ones_like(E_t_label) * (de - 1)

            elif self.rdb_crit == 'p_x1_g_xt':
                x_column_idxs = torch.unsqueeze(torch.argmax(X_1_pred, dim=-1), dim=-1)
                e_column_idxs = torch.unsqueeze(torch.argmax(E_1_pred, dim=-1), dim=-1)

            elif self.rdb_crit == 'x_1':
                x_column_idxs = torch.unsqueeze(X_1_sampled, dim=-1)
                e_column_idxs = torch.unsqueeze(E_1_sampled, dim=-1)

            elif self.rdb_crit == 'p_xt_g_x1':
                x_column_idxs = torch.unsqueeze(torch.argmax(pX, dim=-1), dim=-1)
                e_column_idxs = torch.unsqueeze(torch.argmax(pE, dim=-1), dim=-1)

            elif self.rdb_crit == 'xhat_t':
                sampled_hat = sample_discrete_features(pX, pE, node_mask)
                x_column_idxs = torch.unsqueeze(sampled_hat.X, dim=-1)
                e_column_idxs = torch.unsqueeze(sampled_hat.E, dim=-1)

            else:
                raise NotImplementedError(f"rdb_crit '{self.rdb_crit}' not implemented for column")

            # Create mask: one-hot at column_idx, plus keep current state
            x_col_squeezed = torch.reshape(x_column_idxs, [pX.shape[0], pX.shape[1]])
            e_col_squeezed = torch.reshape(e_column_idxs, pE.shape[:3])

            x_mask = torch.nn.functional.one_hot(x_col_squeezed, num_classes=dx).float()
            e_mask = torch.nn.functional.one_hot(e_col_squeezed, num_classes=de).float()

            # Also keep current state
            eq_x = torch.reshape(x_column_idxs == X_t_label.to(x_mask.shape[:-1] + (1,)),
                            torch.float32)
            eq_e = torch.reshape(e_column_idxs == E_t_label.to(e_mask.shape[:-1] + (1,)),
                            torch.float32)
            x_mask = torch.maximum(x_mask, eq_x)
            e_mask = torch.maximum(e_mask, eq_e)

        elif self.rdb == 'entry':
            if self.rdb_crit == 'abs_state':
                x_masked_idx = dx - 1
                e_masked_idx = de - 1
                x1_idxs = torch.unsqueeze(X_1_sampled, dim=-1)  # (bs, n, 1)
                e1_idxs = torch.unsqueeze(E_1_sampled, dim=-1)  # (bs, n, n, 1)
            elif self.rdb_crit == 'first':
                x_masked_idx = 0
                e_masked_idx = 0
                x1_idxs = torch.unsqueeze(X_1_sampled, dim=-1)
                e1_idxs = torch.unsqueeze(E_1_sampled, dim=-1)
            else:
                raise NotImplementedError(f"rdb_crit '{self.rdb_crit}' not implemented for entry")

            # Build mask via numpy for advanced indexing
            x_mask_np = np.zeros_like(pX.detach().cpu().numpy())
            e_mask_np = np.zeros_like(pE.detach().cpu().numpy())

            X_t_np = X_t_label.detach().cpu().numpy().squeeze(-1).astype(np.int64)
            E_t_np = E_t_label.detach().cpu().numpy().squeeze(-1).astype(np.int64)
            x1_np = x1_idxs.detach().cpu().numpy().squeeze(-1).astype(np.int64)
            e1_np = e1_idxs.detach().cpu().numpy().squeeze(-1).astype(np.int64)

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

            x_mask = torch.tensor(x_mask_np)
            e_mask = torch.tensor(e_mask_np)

        else:
            raise NotImplementedError(f"rdb type '{self.rdb}' not implemented")

        Rdb_X = pX * x_mask * self.eta
        Rdb_E = pE * e_mask * self.eta

        return Rdb_X, Rdb_E

    def _compute_R_tg(self, X_1_sampled, E_1_sampled, X_t_label, E_t_label,
                      dfm_vars):
        """Compute the target guidance rate R^tg."""
        if self.omega == 0:
            zeros_X = torch.zeros_like(dfm_vars['pt_vals_X'])
            zeros_E = torch.zeros_like(dfm_vars['pt_vals_E'])
            return zeros_X, zeros_E

        dx = self.num_classes_X
        de = self.num_classes_E

        X1_oh = torch.nn.functional.one_hot(X_1_sampled, num_classes=dx).float()
        E1_oh = torch.nn.functional.one_hot(E_1_sampled, num_classes=de).float()

        X1_exp = torch.unsqueeze(X_1_sampled, dim=-1)
        mask_X =( X1_exp != X_t_label).to(torch.float32)
        E1_exp = torch.unsqueeze(E_1_sampled, dim=-1)
        mask_E =( E1_exp != E_t_label).to(torch.float32)

        numer_X = X1_oh * self.omega * mask_X
        denom_X = torch.unsqueeze(dfm_vars['Z_X'], dim=-1) * dfm_vars['pt_at_Xt']
        denom_X = torch.where(denom_X == 0, torch.ones_like(denom_X), denom_X)
        Rtg_X = numer_X / denom_X

        numer_E = E1_oh * self.omega * mask_E
        denom_E = torch.unsqueeze(dfm_vars['Z_E'], dim=-1) * dfm_vars['pt_at_Et']
        denom_E = torch.where(denom_E == 0, torch.ones_like(denom_E), denom_E)
        Rtg_E = numer_E / denom_E

        return Rtg_X, Rtg_E

    def _stabilize(self, R_X, R_E, dfm_vars):
        """Post-processing to avoid numerical instabilities."""
        R_X = torch.nan_to_num(R_X, nan=0.0)
        R_E = torch.nan_to_num(R_E, nan=0.0)

        R_X = torch.where(R_X > 1e5, torch.zeros_like(R_X), R_X)
        R_E = torch.where(R_E > 1e5, torch.zeros_like(R_E), R_E)

        pt_at_Xt = dfm_vars['pt_at_Xt']
        zero_mask_X =( pt_at_Xt == 0).to(torch.float32)
        R_X = R_X * (1.0 - zero_mask_X)

        pt_at_Et = dfm_vars['pt_at_Et']
        zero_mask_E =( pt_at_Et == 0).to(torch.float32)
        R_E = R_E * (1.0 - zero_mask_E)

        pX = dfm_vars['pt_vals_X']
        pE = dfm_vars['pt_vals_E']
        col_mask_X =( pX == 0).to(torch.float32)
        R_X = R_X * (1.0 - col_mask_X)
        col_mask_E =( pE == 0).to(torch.float32)
        R_E = R_E * (1.0 - col_mask_E)

        return R_X, R_E


