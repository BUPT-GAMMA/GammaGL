import tensorlayerx as tlx

def cast_bool_to_float(x):
    backend = tlx.BACKEND
    
    if backend == 'tensorflow':
        import tensorflow as tf
        return tf.cast(x, tf.float32)
    elif backend == 'torch':
        import torch
        return x.float()
    elif backend == 'paddle':
        import paddle
        return paddle.cast(x, paddle.float32)
    elif backend == 'mindspore':
        import mindspore as ms
        return ms.ops.Cast()(x, ms.float32)
    else:
        return tlx.cast(x, tlx.float32)

def scatter(input, dim, index, src):
    backend = tlx.BACKEND
    
    if backend == 'tensorflow':
        import tensorflow as tf
        # 创建索引张量
        batch_size = input.shape[0]
        batch_indices = tf.range(batch_size, dtype=tf.int64)
        batch_indices = tf.expand_dims(batch_indices, axis=1)
        
        index_expanded = tf.cast(index, tf.int64)
        indices = tf.concat([batch_indices, index_expanded], axis=1)
        
        return tf.tensor_scatter_nd_update(
            tf.identity(input), 
            indices,
            tf.reshape(src, [-1])
        )
    elif backend == 'torch':
        import torch
        return torch.scatter(input, dim, index, src)
    elif backend == 'paddle':
        import paddle
        return paddle.scatter(input, index, src, axis=dim)
    else:
        return tlx.scatter(input, dim, index, src)

def topk(input, k, dim=None, largest=True, sorted=True):
    backend = tlx.BACKEND
    
    if backend == 'tensorflow':
        import tensorflow as tf
        if dim is None:
            dim = -1
        
        if largest:
            values, indices = tf.math.top_k(input, k=k, sorted=sorted)
        else:
            neg_input = -input
            values, indices = tf.math.top_k(neg_input, k=k, sorted=sorted)
            values = -values
            
        return values, indices
    elif backend == 'torch':
        import torch
        return torch.topk(input, k=k, dim=dim, largest=largest, sorted=sorted)
    elif backend == 'paddle':
        import paddle
        return paddle.topk(input, k=k, axis=dim, largest=largest, sorted=sorted)
    else:
        return tlx.ops.top_k(input, k=k)

def log_softmax(x, dim=None):
    if tlx.BACKEND == 'tensorflow':
        return tlx.ops.log_softmax(x, axis=dim)
    else:
        return tlx.ops.logsoftmax(x, dim=dim)

def gather(input, indices, axis=0):
    if tlx.BACKEND == 'tensorflow':
        return tlx.gather(input, indices, axis=axis)
    else:
        return tlx.gather(input, indices, axis=axis)

def transpose(tensor, perm=None):
    import tensorlayerx as tlx
    
    backend = tlx.BACKEND
    
    if backend == 'torch':
        if perm is None:
            return tensor.t()
        else:
            return tensor.permute(perm)
    elif backend == 'tensorflow':
        import tensorflow as tf
        return tf.transpose(tensor, perm=perm)
    elif backend == 'paddle':
        import paddle
        return paddle.transpose(tensor, perm=perm)
    elif backend == 'mindspore':
        import mindspore as ms
        return ms.ops.Transpose()(tensor, perm)
    else:
        return tlx.ops.transpose(tensor, perm=perm)

def sparse_dense_matmul(indices, values, shape, dense):
    if tlx.BACKEND == 'tensorflow':
        sparse_tensor = tlx.SparseTensor(indices, values, shape)
        return tlx.sparse_tensor_dense_matmul(sparse_tensor, dense)
    else:
        return tlx.sparse_dense_matmul(indices, values, shape, dense)