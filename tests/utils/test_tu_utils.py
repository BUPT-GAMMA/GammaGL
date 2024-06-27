import math
import numpy as np
import tensorlayerx as tlx
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from gammagl.utils.tu_utils import linearsvc,get_negative_expectation,get_positive_expectation,local_global_loss_
import numpy as np
 

def test_tu_utils():
    # Mock data for testing
    l_enc = tlx.convert_to_tensor(np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32))
    g_enc = tlx.convert_to_tensor(np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).astype(np.float32))
    batch = np.array([0, 1, 2]).astype(np.int32)

    # 调试信息
    print("l_enc:", l_enc)
    print("g_enc:", g_enc)
    print("batch:", batch)

    # 计算正负掩码
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = np.zeros((num_nodes, num_graphs), dtype=np.float32)
    neg_mask = np.ones((num_nodes, num_graphs), dtype=np.float32)

    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    print("pos_mask:", pos_mask)
    print("neg_mask:", neg_mask)

    # 转换为张量
    pos_mask_tensor = tlx.convert_to_tensor(pos_mask)
    neg_mask_tensor = tlx.convert_to_tensor(neg_mask)

    # 计算矩阵乘积
    res = tlx.matmul(l_enc, tlx.transpose(g_enc))
    print("res:", res)

    # 计算正负期望
    log_2 = np.log(2)

  
    E_pos = tlx.reduce_sum(get_positive_expectation(res * pos_mask_tensor, average=False))
    E_pos = E_pos / num_nodes
    print("E_pos:", E_pos)

    E_neg = tlx.reduce_sum(get_negative_expectation(res * neg_mask_tensor, average=False))
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    print("E_neg:", E_neg)

    # 计算损失
    loss = E_neg - E_pos
    print("loss:", loss)

    # Test local_global_loss_ function
    expected_loss_value = loss.numpy()  # 计算得到的真实值
    print("Expected loss value:", expected_loss_value)

    # 实际测试
    loss = local_global_loss_(l_enc, g_enc, batch)
    print(loss)
    # Assert statements for testing
    assert np.isclose(loss.numpy(), expected_loss_value, atol=1e-4), f"Expected loss {expected_loss_value} but got {loss.numpy()}"

