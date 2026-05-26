import math
import numpy as np
import tensorlayerx as tlx


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
        t_uniform = tlx.convert_to_tensor(
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
            return (1.0 - tlx.cos(t * math.pi)) / 2.0
        elif distortion_type == 'revcos':
            return 2.0 * t - (1.0 - tlx.cos(t * math.pi)) / 2.0
        elif distortion_type == 'polyinc':
            return t ** 2
        elif distortion_type == 'polydec':
            return 2.0 * t - t ** 2
        else:
            raise ValueError(f"Unknown distortion type: {distortion_type}")
