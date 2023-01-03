import ThrustRTC as trtc

import cupy as cp

perprocess0 = trtc.For(['dst_th', 'indptr_th', 'out_th', 'outdegs_th', 'sample_number'], 'idx',
                       '''
                           outdegs_th[idx] = indptr_th[dst_th[idx] + 1] - indptr_th[dst_th[idx]];
                           if (indptr_th[dst_th[idx] + 1] - indptr_th[dst_th[idx]] > sample_number)
                               out_th[idx] = sample_number;
                           else
                               out_th[idx] = indptr_th[dst_th[idx] + 1] - indptr_th[dst_th[idx]];
                       ''')
perprocess1 = trtc.For(['dst_th', 'indptr_th', 'out_th', ], 'idx',
                       '''
                           out_th[idx] = indptr_th[dst_th[idx] + 1] - indptr_th[dst_th[idx]];
                       ''')

sample_kernel0 = trtc.For(
    ['rng', 'dst_th', 'degs_th', 'sample_cnt_th', 'cnt_ptr_th', 'indptr_th', 'indices_th', 'sample_node_idx_th'],
    'idx',
    '''
    RNGState state;
    // this part why every time the final sample is same?
    rng.state_init(12345, idx, 0, state);
    long long deg = degs_th[idx];
    long long st = cnt_ptr_th[idx];
    long long numb = sample_cnt_th[idx];
    if (deg <= sample_cnt_th[idx])
    {
        for (long long j = 0; j < deg; j++)
            sample_node_idx_th[st + j] = indices_th[indptr_th[dst_th[idx]] + j];
    } else {
        for (long long j = 0; j < numb; j++)
            sample_node_idx_th[st + j] = indices_th[indptr_th[dst_th[idx]] + j];
        for (long long j = numb; j < deg; j++){
            long long cur = (long long)state.rand();
            cur = cur % (j + 1);
            if (cur < numb)
                sample_node_idx_th[st + cur] = indices_th[indptr_th[dst_th[idx]] + j];
        }
    }
    ''')
sample_kernel1 = trtc.For(
    ['dst_th', 'degs_th', 'cnt_ptr_th', 'indptr_th', 'indices_th', 'sample_node_idx_th'],
    'idx',
    '''
    long long deg = degs_th[idx];
    long long st = cnt_ptr_th[idx];
    for (long long j = 0; j < deg; j++)
        sample_node_idx_th[st + j] = indices_th[indptr_th[dst_th[idx]] + j];
    ''')
pre_reindex_code1 = r'''

/*   v          w               x
   node_id,   hash 数值, 原数组中的位置
      y        z
    keys   values   
*/
template <class T>
__device__ bool caninsert(T v, T w, T x, T *y, T *z) {
    T curkey = atomicCAS(reinterpret_cast<unsigned long long int *> (&y[w]), 
                static_cast<unsigned long long int> (-1), static_cast<unsigned long long int> (v));
    if (curkey == -1 || curkey == v) {
        return true;
    }
    return false;
}
template <class T>
__device__ T hash(T x, T y) {
    return x % y;
}

template <class T>
__device__ void insert(T node_idx, T arr_loc, T *keys, T *values, T K) {
    T cur = 1;
    T h = hash(node_idx, K);
    while(!caninsert(node_idx, h, arr_loc, keys, values)){
        h = hash(h + cur, K);
        cur ++;
    }
    keys[h] = node_idx;
    atomicMin(&values[h], arr_loc);
}

extern "C" __global__
void my_func(long long *all_nodes, long long *keys, long long *values, long long K) {
    long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    insert(all_nodes[idx], idx, keys, values, K);
}
'''
pre_reindex1 = cp.RawModule(code=pre_reindex_code1)
pre_reindex_kernel1 = pre_reindex1.get_function("my_func")

pre_reindex_code2 = r'''
template <class T>
__device__ T hash(T x, T y) {
    return x % y;
}

template <class T>
__device__ T search(T node_idx, T *keys, T K) {
    T cur = 1;
    T h = hash(node_idx, K);
    while(keys[h] != node_idx) {
        h = hash(h + cur, K);
        cur += 1;
    }
    return h;
}

extern "C" __global__
void my_func(long long *all_nodes, long long *keys, long long *values, long long *loc, long long K) {
    long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(values[search(all_nodes[idx], keys, K)] == idx)
        loc[idx] = 1;
}
'''
pre_reindex2 = cp.RawModule(code=pre_reindex_code2)
pre_reindex_kernel2 = pre_reindex2.get_function("my_func")

unique_code = r'''
template <class T>
__device__ T hash(T x, T y) {
    return x % y;
}

template <class T>
__device__ T search(T node_idx, T *keys, T K) {
    T cur = 1;
    T h = hash(node_idx, K);
    while(keys[h] != node_idx) {
        h = hash(h + cur, K);
        cur += 1;
    }
    return h;
}
//all_nodes, keys, values, locs, unique_id
extern "C" __global__
void my_func(long long *all_nodes, long long *keys, long long *values, long long *ptr, long long *unique_id, long long K) {
    long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(values[search(all_nodes[idx], keys, K)] == idx){
        unique_id[ptr[idx]] = all_nodes[idx];
    }

}
'''
unique_code_ = cp.RawModule(code=unique_code)
unique_code_kernel = unique_code_.get_function("my_func")

reindex_code = r'''
template <class T>
__device__ T hash(T x, T y) {
    return x % y;
}
template <class T>
__device__ T search(T node_idx, T *keys, T K) {
    T cur = 1;
    T h = hash(node_idx, K);
    while(keys[h] != node_idx){
        h = hash(h + cur, K);
        cur ++;
    }
    return h;
}

extern "C" __global__
void my_func(long long *sample_nodes, long long *keys, long long *values, long long *re, long long K) {
    long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    sample_nodes[idx] = re[values[search(sample_nodes[idx], keys, K)]]; 
}
'''
row_reindex = cp.RawModule(code=reindex_code)
rowreindex_kernel = row_reindex.get_function("my_func")

colreindex_code = r'''

template <class T>
__device__ void cha(T w, T x, T y, T *z) {
    for (T i = 0 ; i < y; i++){
        z[w + i] = x;
    }
}

extern "C" __global__
void my_func(long long *keys, long long *cnt, long long *cntptr, long long *out) {
    long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    cha(cntptr[idx], keys[idx], cnt[idx], out);
}
'''
colreindex = cp.RawModule(code=colreindex_code)
colreindex_kernel = colreindex.get_function("my_func")


class Adj:
    def __init__(self, row, col, size):
        self.row = row
        self.col = col
        self.size = size

    def cp2torch(self):
        import torch
        from torch.utils.dlpack import from_dlpack
        row = from_dlpack(self.row.toDlpack())
        col = from_dlpack(self.col.toDlpack())

        return torch.stack([row, col]), self.size

    def cp2tf(self):
        import tensorflow
        from tensorflow.python import from_dlpack
        row = from_dlpack(self.row.toDlpack())
        col = from_dlpack(self.col.toDlpack())

        return tensorflow.stack([row, col]), self.size