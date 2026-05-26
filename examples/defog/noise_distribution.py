import numpy as np
import tensorlayerx as tlx
from defog_utils import PlaceHolder, backend_one_hot


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
            X=tlx.convert_to_tensor(x_limit),
            E=tlx.convert_to_tensor(e_limit),
            y=tlx.convert_to_tensor(y_limit),
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
            zeros_x = tlx.zeros(list(X.shape[:-1]) + [self.x_added_classes],
                                dtype=X.dtype)
            X = tlx.concat([X, zeros_x], axis=-1)
        if self.e_added_classes > 0:
            zeros_e = tlx.zeros(list(E.shape[:-1]) + [self.e_added_classes],
                                dtype=E.dtype)
            E = tlx.concat([E, zeros_e], axis=-1)
        if y is not None:
            return X, E, y
        return X, E
