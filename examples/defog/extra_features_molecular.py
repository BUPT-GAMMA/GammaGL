import numpy as np
import tensorlayerx as tlx
from defog_utils import PlaceHolder


class ChargeFeature:
    r"""Compute per-node charge using the original DeFoG molecular feature logic.

    Parameters
    ----------
    remove_h : bool
        Whether hydrogens are removed.
    valencies : list
        Expected valency for each atom type.
    """
    def __init__(self, remove_h=True, valencies=None):
        self.remove_h = remove_h
        if valencies is None:
            valencies = [4, 3, 2, 1]
        self.valencies = np.array(valencies, dtype=np.float32)

    def __call__(self, noisy_data):
        X = tlx.convert_to_numpy(noisy_data['X_t'])
        E = tlx.convert_to_numpy(noisy_data['E_t'])
        dx = X.shape[-1]
        de = E.shape[-1]

        if de == 5:
            bond_orders = np.array([0, 1, 2, 3, 1.5], dtype=np.float32).reshape(1, 1, 1, -1)
        else:
            bond_orders = np.array([0, 1, 2, 3], dtype=np.float32).reshape(1, 1, 1, -1)

        weighted_E = E * bond_orders
        current_valencies = np.argmax(weighted_E, axis=-1).sum(axis=-1).astype(np.float32)

        valencies = self.valencies
        if len(valencies) < dx:
            valencies = np.pad(valencies, (0, dx - len(valencies)))
        valencies = valencies.reshape(1, 1, -1)
        weighted_X = X * valencies
        normal_valencies = np.argmax(weighted_X, axis=-1).astype(np.float32)

        charge = (normal_valencies - current_valencies).astype(np.float32)
        return tlx.convert_to_tensor(charge)


class ValencyFeature:
    r"""Compute per-node valency using the original DeFoG molecular feature logic."""
    def __call__(self, noisy_data):
        E = tlx.convert_to_numpy(noisy_data['E_t'])
        de = E.shape[-1]

        if de == 5:
            bond_orders = np.array([0, 1, 2, 3, 1.5], dtype=np.float32).reshape(1, 1, 1, -1)
        else:
            bond_orders = np.array([0, 1, 2, 3], dtype=np.float32).reshape(1, 1, 1, -1)

        weighted_E = E * bond_orders
        valencies = np.argmax(weighted_E, axis=-1).sum(axis=-1).astype(np.float32)
        return tlx.convert_to_tensor(valencies)


class WeightFeature:
    r"""Compute normalized molecular weight.

    Parameters
    ----------
    max_weight : float
        Maximum molecular weight for normalization.
    atom_weights : list or dict
        Atomic weight for each atom type.
    """
    def __init__(self, max_weight=500.0, atom_weights=None):
        self.max_weight = max_weight
        if atom_weights is None:
            atom_weights = [12.0, 14.0, 16.0, 19.0]
        if isinstance(atom_weights, dict):
            atom_weights = [atom_weights[k] for k in sorted(atom_weights.keys())]
        self.atom_weights = np.array(atom_weights, dtype=np.float32)

    def __call__(self, noisy_data):
        X = tlx.convert_to_numpy(noisy_data['X_t'])
        bs, n, dx = X.shape

        atom_types = np.argmax(X, axis=-1)
        aw = self.atom_weights
        if len(aw) < dx:
            aw = np.pad(aw, (0, dx - len(aw)))

        weights = aw[atom_types]
        mol_weight = np.sum(weights, axis=1, keepdims=True).astype(np.float32)
        mol_weight = mol_weight / self.max_weight
        return tlx.convert_to_tensor(mol_weight)


class ExtraMolecularFeatures:
    r"""Molecular-specific extra features: charge, valency, and weight.

    Parameters
    ----------
    dataset_infos : dict
        Dataset information with keys ``'remove_h'``, ``'valencies'``,
        ``'max_weight'``, ``'atom_weights'``.
    """
    def __init__(self, dataset_infos=None):
        if dataset_infos is None:
            dataset_infos = {}
        self.charge = ChargeFeature(
            remove_h=dataset_infos.get('remove_h', True),
            valencies=dataset_infos.get('valencies', None),
        )
        self.valency = ValencyFeature()
        self.weight = WeightFeature(
            max_weight=dataset_infos.get('max_weight', 500.0),
            atom_weights=dataset_infos.get('atom_weights', None),
        )

    def __call__(self, noisy_data):
        """
        Returns
        -------
        PlaceHolder
            X: ``(bs, n, 2)`` (charge + valency), E: empty, y: ``(bs, 1)`` (weight).
        """
        charge = self.charge(noisy_data)    # (bs, n, 1)
        valency = self.valency(noisy_data)  # (bs, n, 1)
        weight = self.weight(noisy_data)    # (bs, 1)

        bs = charge.shape[0]
        n = charge.shape[1]

        return PlaceHolder(
            X=tlx.concat([
                tlx.expand_dims(charge, axis=-1),
                tlx.expand_dims(valency, axis=-1),
            ], axis=-1),
            E=tlx.zeros([bs, n, n, 0], dtype=tlx.float32),
            y=weight,
        )


class DummyMolecularFeatures:
    r"""Dummy molecular features returning empty tensors."""
    def __call__(self, noisy_data):
        X_t = noisy_data['X_t']
        bs = X_t.shape[0]
        n = X_t.shape[1]
        return PlaceHolder(
            X=tlx.zeros([bs, n, 0], dtype=tlx.float32),
            E=tlx.zeros([bs, n, n, 0], dtype=tlx.float32),
            y=tlx.zeros([bs, 0], dtype=tlx.float32),
        )
