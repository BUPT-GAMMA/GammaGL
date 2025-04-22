import tensorlayerx as tlx

def cast_bool_to_float(x):
    """将布尔张量转换为浮点张量"""
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
    """兼容不同后端的 scatter 操作"""
    backend = tlx.BACKEND
    
    if backend == 'tensorflow':
        import tensorflow as tf
        # 创建索引张量
        batch_size = input.shape[0]
        batch_indices = tf.range(batch_size, dtype=tf.int64)
        batch_indices = tf.expand_dims(batch_indices, axis=1)
        
        # 将 batch_indices 和 index 组合成完整的索引
        index_expanded = tf.cast(index, tf.int64)
        indices = tf.concat([batch_indices, index_expanded], axis=1)
        
        # 使用 tf.tensor_scatter_nd_update 更新张量
        return tf.tensor_scatter_nd_update(
            tf.identity(input),  # 创建输入张量的副本
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
        # 默认使用 tlx 的 scatter
        return tlx.scatter(input, dim, index, src)

def topk(input, k, dim=None, largest=True, sorted=True):
    """兼容不同后端的 topk 操作"""
    backend = tlx.BACKEND
    
    if backend == 'tensorflow':
        import tensorflow as tf
        if dim is None:
            dim = -1
        
        if largest:
            values, indices = tf.math.top_k(input, k=k, sorted=sorted)
        else:
            # 对于最小值，先取负，再取最大值，然后再取负
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
        # 默认使用 tlx 的 top_k
        return tlx.ops.top_k(input, k=k)

def log_softmax(x, dim=None):
    """兼容不同后端的 log_softmax 操作"""
    if tlx.BACKEND == 'tensorflow':
        # TensorFlow 使用 axis 而不是 dim
        return tlx.ops.log_softmax(x, axis=dim)
    else:
        return tlx.ops.logsoftmax(x, dim=dim)

def gather(input, indices, axis=0):
    """兼容不同后端的 gather 操作"""
    if tlx.BACKEND == 'tensorflow':
        return tlx.gather(input, indices, axis=axis)
    else:
        # 修改这里，使用 axis 参数而不是 dim
        return tlx.gather(input, indices, axis=axis)

def transpose(tensor, perm=None):
    """
    兼容不同后端的 transpose 操作
    
    Args:
        tensor: 输入张量
        perm: 维度置换顺序
        
    Returns:
        转置后的张量
    """
    import tensorlayerx as tlx
    
    # 获取当前后端
    backend = tlx.BACKEND
    
    if backend == 'torch':
        # PyTorch 后端
        if perm is None:
            return tensor.t()
        else:
            return tensor.permute(perm)
    elif backend == 'tensorflow':
        # TensorFlow 后端
        import tensorflow as tf
        return tf.transpose(tensor, perm=perm)
    elif backend == 'paddle':
        # Paddle 后端
        import paddle
        return paddle.transpose(tensor, perm=perm)
    elif backend == 'mindspore':
        # MindSpore 后端
        import mindspore as ms
        return ms.ops.Transpose()(tensor, perm)
    else:
        # 默认使用 tlx 的 transpose
        return tlx.ops.transpose(tensor, perm=perm)

def sparse_dense_matmul(indices, values, shape, dense):
    """兼容不同后端的稀疏矩阵乘法"""
    if tlx.BACKEND == 'tensorflow':
        sparse_tensor = tlx.SparseTensor(indices, values, shape)
        return tlx.sparse_tensor_dense_matmul(sparse_tensor, dense)
    else:
        # PyTorch 和其他后端
        return tlx.sparse_dense_matmul(indices, values, shape, dense)