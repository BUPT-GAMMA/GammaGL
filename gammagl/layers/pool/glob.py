from typing import Optional
import tensorlayerx as tlx
from gammagl.utils import to_dense_batch

def global_sum_pool(x, batch, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by adding node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n

    Parameters
    ----------
    x: Tensor
        Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
    batch: LongTensor, optional
        Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example.
    size: int, optional
        Batch-size :math:`B`.
        Automatically calculated if not given. (default: :obj:`None`)
    """
    if batch is None:
        return x.sum(dim=0, keepdim=True)
    size = int(tlx.reduce_max(batch) + 1) if size is None else size
    return tlx.unsorted_segment_sum(x, batch, size)


def global_mean_pool(x, batch, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by averaging node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n

    Parameters
    ----------
    x: Tensor
        Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
    batch: LongTensor, optional
        Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example.
    size: int, optional
        Batch-size :math:`B`.
        Automatically calculated if not given. (default: :obj:`None`)
    """
    if batch is None:
        return x.mean(dim=0, keepdim=True)
    size = int(tlx.reduce_max(batch) + 1) if size is None else size
    return tlx.unsorted_segment_mean(x, batch, size)


def global_max_pool(x, batch, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by taking the channel-wise
    maximum across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n

    Parameters
    ----------
        x: Tensor
            Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch: LongTensor, optional
            Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
        size: int, optional
            Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    if batch is None:
        return x.max(dim=0, keepdim=True)[0]
    size = int(tlx.reduce_max(batch) + 1) if size is None else size
    return tlx.unsorted_segment_max(x, batch, size)


def global_min_pool(x, batch, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by taking the channel-wise
    minimum across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \mathrm{min}_{n=1}^{N_i} \, \mathbf{x}_n

    Parameters
    ----------
        x: Tensor
            Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch: LongTensor, optional
            Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
        size: int, optional
            Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    if batch is None:
        return x.min(dim=0, keepdim=True)[0]
    size = int(tlx.reduce_max(batch) + 1) if size is None else size
    return tlx.unsorted_segment_min(x, batch, size)


def global_sort_pool(x, batch, k):
    r"""The global pooling operator from the `"An End-to-End Deep Learning
    Architecture for Graph Classification"
    <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`_ paper,
    where node features are sorted in descending order based on their last
    feature channel. The first :math:`k` nodes form the output of the layer.

    Args:
        x: Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch: Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        k (int): The number of nodes to hold for each graph.

    :rtype: :class:`Tensor`
    """
    fill_value = tlx.reduce_min(x) - 1
    x, _ = to_dense_batch(x, batch, fill_value)
    B, N, D = x.shape

    perm = tlx.argsort(x[:, :, -1], axis=-1, descending=True)
    arange = tlx.arange(0, B) * N
    perm = perm + tlx.reshape(arange, (-1, 1))

    x = tlx.reshape(x, (B * N, -1))
    x = tlx.gather(x, tlx.reshape(perm, (-1, 1)))
    x = tlx.reshape(x, (B, N, -1))

    if N >= k:
        x = x[:, :k]
    else:
        expand_x = tlx.constant(fill_value, dtype=x.dtype, shape=(B, k - N, D), device = x.device)
        x = tlx.concat([x, expand_x], axis=1)

    x = tlx.where(x == fill_value, tlx.zeros_like(x), x)
    x = tlx.reshape(x, (B, -1))

    return x

