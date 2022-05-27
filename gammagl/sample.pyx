# distutils: language = c++



import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdlib cimport rand, srand
from libc.time cimport time
from libc.stdio cimport printf

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_subset(long long k, np.ndarray[np.int64_t, ndim=1] dst_nodes, np.ndarray[np.int64_t, ndim=1] rowptr,
                  np.ndarray[np.int64_t, ndim=2] edge_index, bool replace):
    cdef vector[long long] e_id
    cdef unordered_map[long long, long long] all_nodes
    cdef vector[long long] n_ids
    cdef long long [:,:] edge = edge_index
    cdef unordered_set[long long] cur_eid
    cdef long long i
    for i in xrange(dst_nodes.shape[0]):
        all_nodes[dst_nodes[i]] = i
        n_ids.push_back(dst_nodes[i])
    cdef long long cur
    srand(time(NULL))

    cdef long long j, st, ed
    if k < 0 :
        "full sample"
        with nogil:
            for i in xrange(dst_nodes.shape[0]):
                st = rowptr[dst_nodes[i]]
                ed = rowptr[dst_nodes[i] + 1]
                for j in xrange(st, ed):
                    e_id.push_back(j)
                    if all_nodes.count(edge[j][0]) == 0:
                        all_nodes[edge[j][0]] = n_ids.size()
                        n_ids.push_back(edge[j][0])
    elif replace:
        "sample with replacement"
        with nogil:
            for i in xrange(dst_nodes.shape[0]):
                st = rowptr[dst_nodes[i]]
                ed = rowptr[dst_nodes[i] + 1]
                for j in xrange(st, ed):
                    e_id.push_back(j)
                    if all_nodes.count(edge[j][0]) == 0:
                        all_nodes[edge[j][0]] = n_ids.size()
                        n_ids.push_back(edge[j][0])
    else:

        with nogil:
            for i in xrange(dst_nodes.shape[0]):
                st = rowptr[dst_nodes[i]]
                ed = rowptr[dst_nodes[i] + 1]
                if ed - st > k:
                    '''
                    use rand() of ctime to get over random sample
                    use unordered_set to store current e_id, and accomplish replace == False
                    '''
                    # aa = t.time()
                    for j in xrange(ed - st - k, ed-st):
                        cur = st + (rand() % j)
                        cur_eid.insert(cur)
                else:
                    for j in xrange(st, ed):
                        e_id.push_back(j)
                        if all_nodes.count(edge[j][0]) == 0:
                            all_nodes[edge[j][0]] = n_ids.size()
                            n_ids.push_back(edge[j][0])
    '''
    while replace == false, Cython use function nesting will cost too much time, due to Cython can't 
    define a variable in for loop, so we can put all sample e_id in unordered_set, and process in the end of sample.
    It equal to process each unordered_set per node's sample(PyG's sample_adj method)
    '''
    cdef unordered_set[long long].iterator it = cur_eid.begin()
    while it != cur_eid.end():
        i = deref(it)
        e_id.push_back(i)
        if all_nodes.count(edge[i][0]) == 0:
            all_nodes[edge[i][0]] = n_ids.size()
            n_ids.push_back(edge[i][0])
        inc(it)
    a = dst_nodes.shape[0]
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
