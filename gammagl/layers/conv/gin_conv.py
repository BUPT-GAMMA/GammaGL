import tensorlayerx as tlx
from . import MessagePassing


class GINConv(MessagePassing):
    r"""Graph Isomorphism Network layer from `How Powerful are Graph
    Neural Networks? <https://arxiv.org/pdf/1810.00826.pdf>`__

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    aggregator_type : str
        Aggregator type to use (``sum``, ``max`` or ``mean``).
    init_eps : float, optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter. Default: ``False``.

    Example
    -------
    >>> import numpy as np
    >>> import tensorlayerx as tlx
    >>> from gammagl.layers.conv import GINConv
    >>>
    >>>
    >>> edge_index = np.array([[0,1,2,3,2,5], [1,2,3,4,0,3]])
    >>> x = tlx.ones((6, 10))
    >>> conv = GINConv(aggregator_type='sum')
    >>> res = conv(x, edge_index)
    >>> res
    <tf.Tensor: shape=(6, 10), dtype=float32, numpy=
    array([[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
       [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
       [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]], dtype=float32)>
    >>> tlx.layers.Dense(n_units=10, in_channels=10, name='dense_1')
    >>> conv = GINConv(lin, 'mean')
    >>> res = conv(x, edge_index)
    >>> res
    <tf.Tensor: shape=(6, 10), dtype=float32, numpy=
    array([[-0.0657976 , -0.10810851,  0.19609536,  0.16931635,  0.00967707,
        -0.06709788,  0.04703716,  0.16272973,  0.04323361, -0.1765148 ],
       [-0.0657976 , -0.10810851,  0.19609536,  0.16931635,  0.00967707,
        -0.06709788,  0.04703716,  0.16272973,  0.04323361, -0.1765148 ],
       [-0.0657976 , -0.10810851,  0.19609536,  0.16931635,  0.00967707,
        -0.06709788,  0.04703716,  0.16272973,  0.04323361, -0.1765148 ],
       [-0.0986964 , -0.16216277,  0.29414305,  0.25397453,  0.01451561,
        -0.10064684,  0.07055576,  0.24409458,  0.06485042, -0.2647722 ],
       [-0.0657976 , -0.10810851,  0.19609536,  0.16931635,  0.00967707,
        -0.06709788,  0.04703716,  0.16272973,  0.04323361, -0.1765148 ],
       [-0.0328988 , -0.05405425,  0.09804768,  0.08465818,  0.00483853,
        -0.03354894,  0.02351858,  0.08136486,  0.02161681, -0.0882574 ]],
      dtype=float32)>
    """
    def __init__(self,
                 apply_func=None,
                 aggregator_type='mean',
                 init_eps=0.0,
                 learn_eps=False):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        if aggregator_type in ['mean', 'sum', 'max']:
            self.aggregator_type = aggregator_type
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
        
        # to specify whether eps is trainable or not.
        # self.eps = tlx.Variable(initial_value=[float(init_eps)], name='eps', trainable=learn_eps)
        init = tlx.initializers.Constant(value=init_eps)
        self.eps = self._get_weights(var_name='eps', shape=(1,), init=init)

    def reset_parameters(self):
        pass
    
    def forward(self, x, edge_index):
        r"""Compute Graph Isomorphism Network layer.

        Parameters
        ----------
        x : tlx.Tensor
        edge_index : tlx.Tensor or pair of tf.Tensor
            If a tf.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of tf.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.

        Returns
        -------
        tlx.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        out = self.propagate(x=x, edge_index=edge_index, aggr=self.aggregator_type)

        out += (1 + self.eps) * x

        if self.apply_func is not None:
            out = self.apply_func(out)
        return out
