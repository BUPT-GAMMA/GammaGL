import math
import numpy as np
import tensorlayerx as tlx
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from gammagl.utils.tu_utils import linearsvc,get_negative_expectation,get_positive_expectation,local_global_loss_
import numpy as np


def test_tu_utils():
    l_enc = tlx.convert_to_tensor(np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32))
    g_enc = tlx.convert_to_tensor(np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).astype(np.float32))
    batch = np.array([0, 1, 2]).astype(np.int32)
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]
    pos_mask = np.zeros((num_nodes, num_graphs), dtype=np.float32)
    neg_mask = np.ones((num_nodes, num_graphs), dtype=np.float32)
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.
    pos_mask_tensor = tlx.convert_to_tensor(pos_mask)
    neg_mask_tensor = tlx.convert_to_tensor(neg_mask)
    res = tlx.matmul(l_enc, tlx.transpose(g_enc))
    log_2 = np.log(2)
    E_pos = tlx.reduce_sum(get_positive_expectation(res * pos_mask_tensor, average=False))
    E_pos = E_pos / num_nodes
    E_neg = tlx.reduce_sum(get_negative_expectation(res * neg_mask_tensor, average=False))
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    loss = E_neg - E_pos
    expected_loss_value = loss.numpy() 
    loss = local_global_loss_(l_enc, g_enc, batch)
    assert np.isclose(loss.numpy(), expected_loss_value, atol=1e-4), f"Expected loss {expected_loss_value} but got {loss.numpy()}"

