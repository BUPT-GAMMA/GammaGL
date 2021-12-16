import tensorflow as tf
import tensorlayer as tl


class MessagePassing(tl.layers.Module):
    r"""Base class for creating message passing layers of the form

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    """

    def __init__(self):
        super().__init__()

    def message(self, x):
        return x

    def aggregate(self, inputs, index):
        return tf.math.segment_sum(inputs, index)

    def message_aggregate(self, x, index):
        x = self.message(x)
        x = self.aggregate(x, index)
        return x

    def update(self, x):
        return x

    def propagate(self, x, edge_index):
        out = self.message_aggregate(x, edge_index)
        out = self.update(out)
        return out
