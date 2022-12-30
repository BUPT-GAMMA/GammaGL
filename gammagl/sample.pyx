# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=ascii


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

@cython.boundscheck(False)
@cython.wraparound(False)
def hetero_sample(list node_types, list edge_types, dict row_index_dict, dict rowptrs, dict dstnode_dict,
                           dict num_neighbors_dict, long long num_hops, bool replace, bool directed):
    '''

    :param node_types: ????
    :param edge_types: ??????????[('a','b','c'), ('c','b','a')..]
    :param row_index_dict:
    :param rowptrs: ?????????rowptr {"a__b__c": np.ndarray(ndim=1) ...}
    :param dstnode_dict: ?????? {"a": np.ndarray(ndim=1), "b":.....}
    :param num_neighbors_dict: ??????????{"a__b__c":[10,25], ...}
    :param num_hops: ????
    :param replace:  ????
    :return:
    '''
    # ??????? ????"a__b__c"?????edge_type_name
    # sample_node_type
    cdef string edge_type_name, sample_node_type
    cdef:
        # to_edge_type ??????[('a','b','c')] ==>> ?['a__b__c']??????????src?dst???
        unordered_map[string, vector[string]] to_edge_type
        # ??????edge_dict????????? row??col????????
        unordered_map[string, vector[long long]] edges_dict, rows_dict, cols_dict
        # ??????? ????"a__b__c"?????
    for edge_type in edge_types:
        # to_edge_type['a__b__c'] = ('a','b','c') ...
        edge_type_name = (edge_type[0] + "__" + edge_type[1] + "__" + edge_type[2]).encode("utf-8")
        to_edge_type[edge_type_name].push_back(edge_type[0].encode("utf-8"))
        to_edge_type[edge_type_name].push_back(edge_type[1].encode("utf-8"))
        to_edge_type[edge_type_name].push_back(edge_type[2].encode("utf-8"))
    # ??????????
    for key in rowptrs.keys():
        edges_dict[key.encode("utf-8")]
        rows_dict[key.encode("utf-8")]
        cols_dict[key.encode("utf-8")]

    # ?????????????????
    cdef:
        # ???????node, samples_dict['a'] : ???node_id
        unordered_map[string, vector[long long]] samples_dict
        # ????????????????
        unordered_map[string, unordered_map[long long, long long]] to_local_node_dict
    for node_type in node_types:
        samples_dict[node_type.encode("utf-8")]
        to_local_node_dict[node_type.encode('utf-8')]

    # =============================  ??????????????c++????????????????? =================================================#
    cdef:
        vector[long long] *samples
        unordered_map[long long, long long] *to_local_node
        pair[long long, long long] xxxx
        long long i
        long long length
        long long node
    #   ?dstnode_dict??????????????????
    for samples_key in dstnode_dict:
        # ??????????????
        # samples = &samples_dict[samples_key.encode('utf-8')]
        # to_local_node = &to_local_node_dict[samples_key.encode('utf-8')]
        length = dstnode_dict[samples_key].shape[0]
        for i in xrange(length):
            samples_dict[samples_key.encode('utf-8')].push_back(dstnode_dict[samples_key][i])
            xxxx.first = dstnode_dict[samples_key][i]
            xxxx.second = i
            to_local_node_dict[samples_key.encode('utf-8')].insert(xxxx)
            # samples.push_back(dstnode_dict[samples_key][i])
            # xxxx.first = dstnode_dict[samples_key][i]
            # xxxx.second = i
            # to_local_node.insert(xxxx)

    # ?????????
    cdef:
        unordered_map[string, pair[long long, long long]] st_ed_dict
        unordered_map[string, vector[long long]].iterator it
        pair[long long, long long] xx

    it = samples_dict.begin()
    while it != samples_dict.end():
        st_ed_dict[deref(it).first].first = 0
        st_ed_dict[deref(it).first].second = deref(it).second.size()
        inc(it)

    # ======================  k ? ???????? ===========================
    cdef:
        long long j, k, l, v, u
        vector[string] a__b__c
        string src_node_type, dst_node_type
        long long sample_number, start, end, cur_node, colst, coled, colcnt
        vector[long long] dst_samples
        vector[long long] *edge_dict
        vector[long long] *col_dict
        vector[long long] *row_dict
        vector[long long].iterator vit
        # unordered_map[long long, long long] *to_local_src_node
        pair[long long, long long] xxx
        long long [:] indptr, col, row
        pair[unordered_map[long long, long long].iterator, bool] insertt
    # ????????????
    cdef:
        unordered_map[long long, unordered_set[long long]] rnd_set
        unordered_set[long long] sett
    rnd_set[0]
    # ======================  k ? ????  ===========================
    srand(time(NULL))
    for i in xrange(num_hops):
        for num_neighbors_key in num_neighbors_dict:
            # ???????????a,c

            edge_type_name = num_neighbors_key.encode("utf-8")
            a__b__c = to_edge_type[edge_type_name]

            src_node_type = a__b__c[0]
            dst_node_type = a__b__c[2]

            # ????????src\dst???id
            sample_number = num_neighbors_dict[num_neighbors_key][i]
            dst_samples = samples_dict.at(dst_node_type)


            edge_dict = &edges_dict.at(edge_type_name)
            col_dict = &cols_dict.at(edge_type_name)
            row_dict = &rows_dict.at(edge_type_name)


            indptr = rowptrs[num_neighbors_key]
            row = row_index_dict[num_neighbors_key]

            start = st_ed_dict.at(dst_node_type).first
            end = st_ed_dict.at(dst_node_type).second


            # ===============================  ??????? =================================================

            for j in xrange(start, end):
                v = dst_samples[j]
                # v = deref(vit + j)
                # print("v:", v)
                coled = indptr[v+1]
                colst = indptr[v]
                colcnt = coled - colst
                if colcnt == 0:
                    continue
                if sample_number < 0:
                    for k in xrange(colst, coled):
                        u = row[k]
                        xxx.first = u
                        xxx.second = samples_dict.at(src_node_type).size()
                        # c++??[iteratir, bool] ? pair
                        insertt = to_local_node_dict.at(src_node_type).insert(xxx)
                        if insertt.second:
                            samples_dict.at(src_node_type).push_back(u)

                        if directed:
                            col_dict.push_back(j)
                            row_dict.push_back(deref(insertt.first).second)
                            edge_dict.push_back(k)
                elif replace:
                    for k in xrange(sample_number):
                        # ??????
                        l = colst + (rand() % colcnt)
                        u = row[l]
                        xxx.first = u
                        xxx.second = samples_dict.at(src_node_type).size()
                        # c++??[iteratir, bool] ? pair
                        insertt = to_local_node_dict.at(src_node_type).insert(xxx)
                        if insertt.second:
                            samples_dict.at(src_node_type).push_back(u)
                        if directed:
                            col_dict.push_back(j)
                            row_dict.push_back(deref(insertt.first).second)
                            edge_dict.push_back(l)
                else:
                    # ?????????????????
                    if sample_number >= colcnt:
                        for k in xrange(colst, coled):
                            u = row[k]
                            xxx.first = u
                            xxx.second = samples_dict.at(src_node_type).size()
                            # c++??[iteratir, bool] ? pair
                            insertt = to_local_node_dict.at(src_node_type).insert(xxx)
                            if insertt.second:
                                samples_dict.at(src_node_type).push_back(u)

                            if directed:
                                col_dict.push_back(j)
                                row_dict.push_back(deref(insertt.first).second)
                                edge_dict.push_back(k)
                    else:
                        # ??????set??????set?????????c++?????
                        sett = rnd_set[0]
                        for k in xrange(colcnt - sample_number, colcnt):
                            l = rand() % k
                            # ??????
                            if not sett.insert(l).second:
                                l = k
                                sett.insert(k)
                            l += colst
                            u = row[l]
                            xxx.first = u
                            xxx.second = samples_dict.at(src_node_type).size()
                            insertt = to_local_node_dict.at(src_node_type).insert(xxx)
                            if insertt.second:
                                samples_dict.at(src_node_type).push_back(u)

                            if directed:
                                col_dict.push_back(j)
                                row_dict.push_back(deref(insertt.first).second)
                                edge_dict.push_back(k)

        # ???????????

        it = samples_dict.begin()
        while it != samples_dict.end():
            st_ed_dict[deref(it).first].first = st_ed_dict[deref(it).first].second
            st_ed_dict[deref(it).first].second = deref(it).second.size()
            inc(it)
        # print("yici caiyang")

    return samples_dict, rows_dict, cols_dict, edges_dict
# python setup.py build_ext  --inplace