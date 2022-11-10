import torch

use_ext = False
try:
    import torch_segment
    import torch_gspmm
    use_ext = True
except:
    pass


def unsorted_segment_sum(x, segment_ids, num_segments=None):
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    if num_segments is None:
        num_segments = int(segment_ids.max().item()) + 1
    # else:
        # `rgcn` meet an error that `segment_ids` is empty
        # assert segment_ids.max() < num_segments


    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(x.shape[1:], device=x.device)).to(torch.int64)
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *x.shape[1:])

    assert x.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(x.shape[1:])
    tensor = torch.zeros(*shape, device=x.device).to(x.dtype).scatter_add(0, segment_ids, x)
    return tensor


def unsorted_segment_mean(x, segment_ids, num_segments=None):
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    if num_segments is None:
        num_segments = int(segment_ids.max().item()) + 1
    # else:
    # `rgcn` meet an error that `segment_ids` is empty
    # assert segment_ids.max() < num_segments

    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(x.shape[1:], device=x.device)).to(torch.int64)
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *x.shape[1:])

    assert x.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(x.shape[1:])
    ones_data = torch.ones_like(x, dtype=x.dtype, device=x.device)
    tensor = torch.zeros(*shape, device=x.device).to(x.dtype).scatter_add(0, segment_ids, x)
    tensor_nums = torch.zeros(*shape, device=x.device).to(x.dtype).scatter_add(0, segment_ids, ones_data)
    tensor = tensor / tensor_nums
    tensor[torch.isnan(tensor)] = 0
    return tensor


def unsorted_segment_max(x, segment_ids, num_segments=None):
    if num_segments is None:
        num_segments = int(segment_ids.max().item()) + 1
    # else:
    # `rgcn` meet an error that `segment_ids` is empty
    # assert segment_ids.max() < num_segments

    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    if use_ext:
        return torch_segment.segment_max(x, segment_ids, num_segments)
    res = []
    for i in range(num_segments):
        res.append(torch.max(x[segment_ids == i], dim=0)[0])
    return torch.stack(res, dim=0)


def segment_max(x, segment_ids, num_segments=None):
    return unsorted_segment_max(x, segment_ids,num_segments)


def segment_mean(x, segment_ids,num_segments=None):
    return unsorted_segment_mean(x, segment_ids,num_segments)


def segment_sum(x, segment_ids,num_segments=None):
    return unsorted_segment_sum(x, segment_ids,num_segments)

def gspmm(index, weight, x, reduce='sum'):
    if reduce == 'sum':
        return torch_gspmm.spmm_sum(index, weight, x)
    else:
        raise Exception("Unsupported reduce type, please choose from ['sum', ].")
