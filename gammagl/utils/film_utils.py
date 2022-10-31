import tensorlayerx as tlx

TF_BACKEND = 'tensorflow'
TORCH_BACKEND = 'torch'
PADDLE_BACKEND = 'paddle'
MS_BACKEND = 'mindspore'


def split_to_two(tensor, axis=-1):
    if tlx.BACKEND == TORCH_BACKEND or tlx.BACKEND == MS_BACKEND:
        return tlx.split(tensor, tensor.shape[-1] // 2, axis)
    elif tlx.BACKEND == TF_BACKEND:
        return tlx.split(tensor, 2, axis)
    elif tlx.BACKEND == PADDLE_BACKEND:
        return tensor.split(2, axis=axis)


def product(xs, ys):
    res = []
    for x in xs:
        for y in ys:
            res.append((x, y))
    return res
