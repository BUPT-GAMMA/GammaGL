# from gammagl.datasets import Reddit
#
# dataset = Reddit("./reddit/", "Reddit")
# g = dataset.data
# print(g.edge_index)
import tensorlayerx as tlx
import numpy as np
# # 代表src、dst、weight
# a = np.array([[1,0,3,4,5],
#               [2,1,3,0,1],
#               [0.1,0.2,0.3,0.4,0.5]])
# # 对dst排序
# ind = np.array(a[1], dtype=np.int32)
# ind = np.argsort(ind, axis=0)
# print(ind)
# # 输出排序后的边
# print((a.T[ind]).T)


a = tlx.convert_to_tensor(np.arange(0,100), )


print(a)