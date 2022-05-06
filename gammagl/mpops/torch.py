import torch


def unsorted_segment_sum(x, segment_ids, num_segments=None):
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    # if num_segments is not None:
    #     # `rgcn` meet an error that `segment_ids` is empty
    #     assert segment_ids.max() < num_segments

    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(x.shape[1:])).to(torch.int32)
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *x.shape[1:])

    assert x.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(x.shape[1:])
    tensor = torch.zeros(*shape).to(x.dtype).scatter_add(0, segment_ids, x)
    return tensor


def unsorted_segment_mean(x, segment_ids, num_segments=None):
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    if num_segments is not None:
        assert segment_ids.max() < num_segments

    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(x.shape[1:])).to(torch.int32)
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *x.shape[1:])

    assert x.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(x.shape[1:])
    ones_data = torch.ones_like(x, dtype=x.dtype)
    tensor = torch.zeros(*shape).to(x.dtype).scatter_add(0, segment_ids, x)
    tensor_nums = torch.zeros(*shape).to(x.dtype).scatter_add(0, segment_ids, ones_data)
    tensor = tensor / tensor_nums
    return tensor


def unsorted_segment_max(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert segment_ids.max() < num_segments

    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    res = []
    for i in range(num_segments):
        res.append(torch.max(x[segment_ids == i], dim=0)[0])
    return torch.stack(res, dim=0)


def segment_max(x, segment_ids):
    return unsorted_segment_max(x, segment_ids)


def segment_mean(x, segment_ids):
    return unsorted_segment_mean(x, segment_ids)


def segment_sum(x, segment_ids):
    return unsorted_segment_sum(x, segment_ids)

