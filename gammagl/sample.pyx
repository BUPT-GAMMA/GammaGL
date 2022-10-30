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


cdef extern from "stdint.h":
    ctypedef signed int int64_t


cdef extern from *:
    """
    #if defined(_WIN32) || defined(MS_WINDOWS) || defined(_MSC_VER)
        #include "third_party/metis/include/fake.h"
        #define win32 1
        #define METIS_Recursive_(a,b,c,d,e,f,g,h,i,j,k,l,m) fake_function_0(a,b,c,d,e,f,g,h,i,j,k,l,m)
        #define METIS_Kway_(a,b,c,d,e,f,g,h,i,j,k,l,m) fake_function_1(a,b,c,d,e,f,g,h,i,j,k,l,m)
    #else
        #include "third_party/metis/include/metis.h"
        #define win32 0
        #define METIS_Recursive_(a,b,c,d,e,f,g,h,i,j,k,l,m) METIS_PartGraphRecursive(a,b,c,d,e,f,g,h,i,j,k,l,m)
        #define METIS_Kway_(a,b,c,d,e,f,g,h,i,j,k,l,m) METIS_PartGraphKway(a,b,c,d,e,f,g,h,i,j,k,l,m)

    #endif
    """
    bool win "win32"
    int METIS_Recursive "METIS_Recursive_"(int64_t *nvtxs, int64_t *ncon, int64_t *xadj,
                  int64_t *adjncy, int64_t *vwgt, int64_t *vsize, int64_t *adjwgt,
                  int64_t *nparts, float *tpwgts, float *ubvec, int64_t *options,
                  int64_t *edgecut, int64_t *part) nogil
    int METIS_Kway "METIS_Kway_"(int64_t *nvtxs, int64_t *ncon, int64_t *xadj,
                  int64_t *adjncy, int64_t *vwgt, int64_t *vsize, int64_t *adjwgt,
                  int64_t *nparts, float *tpwgts, float *ubvec, int64_t *options,
                  int64_t *edgecut, int64_t *part) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
def sample_subset(int64_t k, np.ndarray[np.int64_t, ndim=1] dst_nodes, np.ndarray[np.int64_t, ndim=1] rowptr,
                  np.ndarray[np.int64_t, ndim=2] edge_index, bool replace):
    cdef vector[int64_t] e_id
    cdef unordered_map[int64_t, int64_t] all_nodes
    cdef vector[int64_t] n_ids
    cdef int64_t [:,:] edge = edge_index
    cdef unordered_set[int64_t] cur_eid
    cdef int64_t i
    for i in xrange(dst_nodes.shape[0]):
        all_nodes[dst_nodes[i]] = i
        n_ids.push_back(dst_nodes[i])
    cdef int64_t cur
    srand(time(NULL))

    cdef int64_t j, st, ed
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
    cdef unordered_set[int64_t].iterator it = cur_eid.begin()
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
    cdef int64_t num_e = e_id.size()
    cdef np.ndarray[np.int64_t, ndim=2] smallg = np.empty([num_e, 2], dtype=np.int64)
    cdef int64_t [:, :] eind = smallg
    with nogil:
        for i in xrange(num_e):
            eind[i][0] = all_nodes[edge[e_id[i]][0]]
            eind[i][1] = all_nodes[edge[e_id[i]][1]]
    return all_node, (b, a), smallg



@cython.boundscheck(False)
@cython.wraparound(False)
def metis_partition(
    np.ndarray[np.int64_t, ndim=1] indptr,
    np.ndarray[np.int64_t, ndim=1] col,
    int64_t nparts,
    np.ndarray[np.int64_t, ndim=1] node_weights=None,
    np.ndarray[np.int64_t, ndim=1] edge_weights=None,
    bool recursive=True,
):
    cdef:
        int64_t nvtxs = indptr.shape[0] - 1
        int64_t objval = -1
        int64_t ncon = 1
        np.ndarray part = np.zeros((nvtxs, ), dtype="int64")
        int64_t * node_weight_ptr = NULL
        int64_t * edge_weight_ptr = NULL

    if node_weights is not None:
        node_weight_ptr = <int64_t *> node_weights.data
    if edge_weights is not None:
        edge_weight_ptr = <int64_t *> edge_weights.data


    if win == 0:
        with nogil:
            if recursive:
                METIS_Recursive(nvtxs=&nvtxs, ncon=&ncon, xadj=<int64_t *> indptr.data,
                             adjncy=<int64_t *> col.data, vwgt=node_weight_ptr, vsize=NULL, adjwgt=edge_weight_ptr,
                             nparts=&nparts, tpwgts=NULL, ubvec=NULL, options=NULL,
                             edgecut=&objval, part=<int64_t *> part.data)
            else:
                METIS_Kway(nvtxs=&nvtxs, ncon=&ncon, xadj=<int64_t *> indptr.data,
                             adjncy=<int64_t *> col.data, vwgt=node_weight_ptr, vsize=NULL, adjwgt=edge_weight_ptr,
                             nparts=&nparts, tpwgts=NULL, ubvec=NULL, options=NULL,
                             edgecut=&objval, part=<int64_t *> part.data)
    else:
        return -1
    return part
@cython.boundscheck(False)
@cython.wraparound(False)
def ind2ptr(int64_t M,
            np.ndarray[np.int64_t, ndim=1] ind
):
    cdef:
        np.ndarray[np.int64_t, ndim=1] out = np.zeros([M + 1], dtype=np.int64)
        int64_t i = 0, j = 0
        int64_t numel = ind.shape[0]
    if numel == 0:
        return out
    for i in xrange(1, numel):
        for j in xrange(ind[i - 1], ind[i]):
            out[j + 1] = i;
    # for (int64_t i = ind_data[numel - 1] + 1; i < M + 1; i++)
    # out_data[i] = numel;
    for i in xrange(ind[numel - 1] + 1, M + 1):
        out[i] = numel
    return out