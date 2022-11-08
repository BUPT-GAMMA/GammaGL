import tensorlayerx as tlx

TF_BACKEND = 'tensorflow'
TORCH_BACKEND = 'torch'
PADDLE_BACKEND = 'paddle'
MS_BACKEND = 'mindspore'


def product(xs, ys):
    res = []
    for x in xs:
        for y in ys:
            res.append((x, y))
    return res
