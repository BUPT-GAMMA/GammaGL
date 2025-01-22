import jittor as jt


def unsorted_segment_sum(x, segment_ids, num_segments):
    if num_segments is None:
        num_segments = int(segment_ids.asnumpy().max() + 1)
        
    segment_ids = jt.array(segment_ids, dtype=jt.int64)
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    if len(segment_ids.shape) == 1:
        s = jt.prod(jt.array(tuple(x.shape[1:]))).to(jt.int32).item()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *x.shape[1:])
        
    assert x.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(x.shape[1:])
    tensor = jt.zeros(*shape).to(x.dtype).scatter(0, segment_ids, x, 'add')
    return tensor

def unsorted_segment_mean(x, segment_ids, num_segments=None):
    if num_segments is None:
        num_segments = int(segment_ids.numpy().max() + 1)
        
    segment_ids = jt.array(segment_ids, dtype=jt.int64)
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    res = []
    for i in range(num_segments):
        mask_index = segment_ids == i
        if jt.any(mask_index):
            a = jt.mean(x[mask_index], 0)
            res.append(a)
        else:
            a = jt.zeros_like(x[0])
            res.append(a)
    if res[0].shape == [1]:
        return jt.concat(res, 0)
    else:
        return jt.stack(res, 0)
    
def unsorted_segment_max(x, segment_ids, num_segments=None):
    if num_segments is None:
        num_segments = int(segment_ids.numpy().max() + 1)
        
    segment_ids = jt.array(segment_ids, dtype=jt.int64)
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    res = []
    for i in range(num_segments):
        mask_index = segment_ids == i
        if jt.any(mask_index):
            res.append(jt.max(x[mask_index], 0)[0])
        else:
            a = jt.zeros_like(x[0])
            a.fill_(jt.array(float('-inf')).to(a.dtype))
            res.append(a)
    if res[0].shape == [1]:
        return jt.concat(res, 0)
    else:
        return jt.stack(res, 0)
    

def segment_sum(x, segment_ids, num_segments=None):
    if num_segments is None:
        num_segments = int(segment_ids.numpy().max() + 1)
        
    segment_ids = jt.array(segment_ids, dtype=jt.int64)
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    if len(segment_ids.shape) == 1:
        s = jt.prod(jt.array(x.shape[1:])).to(jt.int32)
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *x.shape[1:])

    assert x.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(x.shape[1:])
    tensor = jt.zeros(*shape).to(x.dtype).scatter_add(0, segment_ids, x)
    return tensor



def segment_mean(x, segment_ids, num_segments=None):
    if num_segments is None:
        num_segments = int(segment_ids.numpy().max() + 1)
        
    segment_ids = jt.array(segment_ids, dtype=jt.int64)
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    res = []
    for i in range(num_segments):
        mask_index = segment_ids == i
        if jt.any(mask_index):
            a = jt.mean(x[mask_index], 0)
            res.append(a)
        else:
            a = jt.zeros_like(x[0])
            res.append(a)
    if res[0].shape == [1]:
        return jt.concat(res, 0)
    else:
        return jt.stack(res, 0)
    
def segment_max(x, segment_ids, num_segments=None):
    if num_segments is None:
        num_segments = int(segment_ids.numpy().max() + 1)
        
    segment_ids = jt.array(segment_ids, dtype=jt.int64)
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    res = []
    for i in range(num_segments):
        mask_index = segment_ids == i
        if jt.any(mask_index):
            res.append(jt.max(x[mask_index], 0)[0])
        else:
            a = jt.zeros_like(x[0])
            a.fill_(jt.array(float('-inf')).to(a.dtype))
            res.append(a)
    if res[0].shape == [1]:
        return jt.concat(res, 0)
    else:
        return jt.stack(res, 0)

def gspmm(index, weight=None, x=None, reduce='sum'):
    pass

def bspmm(index, weight=None, x=None, reduce='sum'):
    pass