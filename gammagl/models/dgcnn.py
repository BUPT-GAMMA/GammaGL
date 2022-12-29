import copy

import tensorlayerx as tlx
import tensorlayerx.nn as nn
import numpy as np



def knn(x, k):
    inner = -2*tlx.bmm(tlx.transpose(x, (0, 2, 1)), x)
    xx = tlx.reduce_sum(x**2, axis=1, keepdims=True)
    pairwise_distance = xx + inner + tlx.transpose(xx, (0, 2, 1))

    # idx = pairwise_distance.topk(k=k, dim=-1)[1]  # indices
    # top k indices
    pd_numpy = tlx.convert_to_numpy(pairwise_distance)
    partition_index = np.take(np.argpartition(pd_numpy, kth=k, axis=-1),
                              range(0, k), -1)
    top_scores = np.take_along_axis(pd_numpy, partition_index, -1)
    sorted_index = np.argsort(top_scores, axis=-1)
    idx_numpy = np.take_along_axis(partition_index, sorted_index, -1)
    idx = tlx.convert_to_tensor(idx_numpy)

    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.shape[0]
    num_points = x.shape[2]
    x = tlx.reshape(x, (batch_size, -1, num_points))
    num_dims = x.shape[1]
    if idx is None:
        idx = knn(x, k=k)
    # device = tlx.set_device('CPU')
    idx_base = tlx.reshape(tlx.arange(0, limit=batch_size, dtype=tlx.int64), (-1, 1, 1)) * num_points
    idx = idx + idx_base
    idx = tlx.reshape(idx, (-1,))
    x = tlx.transpose(x, (0, 2, 1))
    feature = tlx.gather(tlx.reshape(x, (batch_size*num_points, -1)), idx, axis=0)
    feature = tlx.reshape(feature, (batch_size, num_points, k, num_dims))
    x = tlx.tile(tlx.reshape(x, (batch_size, num_points, 1, num_dims)), [1, 1, k, 1])
    feature = tlx.concat([feature - x, x], axis=3)
    return feature


class DGCNNModel(nn.Module):
    r"""The Edge Convolution operator from the `"Dynamic Graph CNN for Learning on Point Clouds"
    <https://arxiv.org/pdf/1801.07829.pdf>`_ paper

    """
    def __init__(self, in_channels, k, emb_dims, num_points, dropout, output_channels=40):
        super(DGCNNModel, self).__init__()
        self.k = k
        self.in_channels = in_channels
        self.num_points = num_points

        self.bn1 = nn.BatchNorm2d(num_features=64, momentum=0.1)
        self.bn2 = nn.BatchNorm2d(num_features=64, momentum=0.1)
        self.bn3 = nn.BatchNorm2d(num_features=128, momentum=0.1)
        self.bn4 = nn.BatchNorm2d(num_features=256, momentum=0.1)
        self.bn5 = nn.BatchNorm1d(num_features=emb_dims, momentum=0.1)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=2 * in_channels, out_channels=64, kernel_size=1, b_init=None),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Transpose(perm=[0, 3, 1, 2]))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64*2, out_channels=64, kernel_size=1, b_init=None),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Transpose(perm=[0, 3, 1, 2]))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64 * 2, out_channels=128, kernel_size=1, b_init=None),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Transpose(perm=[0, 3, 1, 2]))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=128 * 2, out_channels=256, kernel_size=1, b_init=None),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Transpose(perm=[0, 3, 1, 2]))
        self.conv5 = nn.Sequential(nn.Conv1d(in_channels=512, out_channels=emb_dims, kernel_size=1, b_init=None),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Transpose(perm=[0, 1, 2]))
        self.linear1 = nn.Linear(in_features=emb_dims*2, out_features=512, b_init=None)
        self.bn6 = nn.BatchNorm1d(num_features=512, momentum=0.1)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(in_features=512, out_features=256)
        self.bn7 = nn.BatchNorm1d(num_features=256, momentum=0.1)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(in_features=256, out_features=output_channels)

    def forward(self, x):
        x = tlx.reshape(x, (-1, self.num_points, self.in_channels))
        x = tlx.transpose(x, perm=(0, 2, 1))
        batch_size = x.shape[0]
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = tlx.reduce_max(x, axis=-1, keepdims=False)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = tlx.reduce_max(x, axis=-1, keepdims=False)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = tlx.reduce_max(x, axis=-1, keepdims=False)

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = tlx.reduce_max(x, axis=-1, keepdims=False)

        x = tlx.concat([x1, x2, x3, x4], axis=1)
        x = self.conv5(tlx.transpose(x, (0, 2, 1)))

        x1 = tlx.nn.AdaptiveMaxPool1d(output_size=1)(x)
        x1 = tlx.reshape(x1, (batch_size, -1))
        x2 = tlx.nn.AdaptiveAvgPool1d(output_size=1)(x)
        x2 = tlx.reshape(x2, (batch_size, -1))
        x = tlx.concat([x1, x2], axis=1)

        x = tlx.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = tlx.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

