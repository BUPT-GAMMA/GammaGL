# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython

from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
from libc.stdlib cimport rand, srand
from libc.time cimport time


@cython.boundscheck(False)
@cython.wraparound(False)
def sample_subset(long long k, np.ndarray[np.int64_t, ndim=1] dst_nodes, np.ndarray[np.int64_t, ndim=1] rowptr,
                  np.ndarray[np.int64_t, ndim=2] edge_index):
    '''

    :param k: 采样个数
    :param dst_nodes: 目的节点
    :param rowptr: 矩阵的rowptr
    :param edge_index: 矩阵
    :return:
    '''


    cdef vector[long long ] e_id # 返回的边索引

    cdef unordered_map[long long, long long] all_nodes #返回的key为所有采样节点，value为生成小图所需的数值
    cdef vector[long long] n_ids # 存储采样节点
    cdef long long [:,:] edge = edge_index

    cdef long long i
    # 压入vector，同时hashmap作为小图的生成

    for i in xrange(dst_nodes.shape[0]):
        all_nodes[dst_nodes[i]] = i
        n_ids.push_back(dst_nodes[i])

    cdef a, b
    a = n_ids.size()
    cdef long long st, ed
    cdef long long j, g, cur
    cdef unordered_set[long long] cur_eid
    srand(time(NULL))

    if k < 0:
        for i in xrange(dst_nodes.shape[0]):
            st = rowptr[dst_nodes[i]]
            ed = rowptr[dst_nodes[i] + 1]
            with nogil:
                for j in xrange(st, ed):
                    e_id.push_back(j)
                    if not all_nodes.count(edge[j][0]):
                        all_nodes[edge[j][0]] = n_ids.size()
                        n_ids.push_back(edge[j][0])

    else:

        with nogil:
            for i in xrange(dst_nodes.shape[0]):
                st = rowptr[dst_nodes[i]]
                ed = rowptr[dst_nodes[i] + 1]
                if ed - st > k:
                    '''该怎么实现随机？nogil下无法实现python类的生成,使用c中的rand'''
                    for j in xrange(ed - st - k, ed-st):
                        cur = st + (rand() % j)
                        # printf("rand: %lld", rand())
                        # if not cur_eid.insert(cur).second:
                        cur_eid.insert(cur)
                        e_id.push_back(cur)
                        if not all_nodes.count(edge[cur][0]):
                            all_nodes[edge[cur][0]] = n_ids.size()
                            n_ids.push_back(edge[cur][0])
                else:
                    for j in xrange(st, ed):

                        e_id.push_back(j)
                        if not all_nodes.count(edge[j][0]):
                            all_nodes[edge[j][0]] = n_ids.size()
                            n_ids.push_back(edge[j][0])


    b = n_ids.size()


    cdef np.ndarray[np.int64_t, ndim=1] all_node = np.empty([n_ids.size()], dtype=np.int64)
    for i in xrange(n_ids.size()):
        all_node[i] = n_ids[i]


    cdef long long num_e = e_id.size()
    cdef np.ndarray[np.int64_t, ndim=2] smallg = np.empty([num_e, 2], dtype=np.int64)
    cdef long long [:, :] eind = smallg
    with nogil:
        for i in xrange(num_e):
            eind[i][0] = all_nodes[edge[e_id[i]][0]]
            eind[i][1] = all_nodes[edge[e_id[i]][1]]

    return all_node, (b, a), smallg









