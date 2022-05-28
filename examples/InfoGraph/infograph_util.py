import tensorlayerx as tlx
import numpy as np
import math


def get_positive_expectation(p_samples, average=True):  # 正样本计算
    """Computes the positive part of a JS Divergence.
    Args:
        p_samples: Positive samples.
        average: Average the result over samples.
    Returns:
        th.Tensor
    """
    log_2 = math.log(2.)
    Ep = log_2 - tlx.softplus(-p_samples)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, average=True):  # 负样本计算
    """Computes the negative part of a JS Divergence.
    Args:
        q_samples: Negative samples.
        average: Average the result over samples.
    Returns:
        th.Tensor
    """
    log_2 = math.log(2.)
    Eq = tlx.softplus(-q_samples) + q_samples - log_2
    if average:
        return Eq.mean()
    else:
        return Eq



def local_global_loss_(l_enc, g_enc, batch):
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = np.zeros((num_nodes, num_graphs), dtype=np.float32)
    neg_mask = np.ones((num_nodes, num_graphs), dtype=np.float32)

    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = tlx.matmul(l_enc, tlx.transpose(g_enc))

    E_pos = tlx.reduce_sum(get_positive_expectation(res * tlx.convert_to_tensor(pos_mask), average=False))
    E_pos = E_pos / num_nodes
    E_neg = tlx.reduce_sum(get_negative_expectation(res * tlx.convert_to_tensor(neg_mask), average=False))
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos
