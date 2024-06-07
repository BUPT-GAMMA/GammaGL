import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['TL_BACKEND'] = 'paddle'
import tensorlayerx as tlx
from gammagl.datasets.acm4heco import ACM4HeCo

root = './data'

def acm():
    dataset = ACM4HeCo(root)
    graph = dataset[0]
    assert len(dataset) == 1
    assert tlx.get_tensor_shape(graph['pos_set_for_contrast']) == [4019, 4019]
    assert len(graph['paper'].nei) == 2
    assert tlx.get_tensor_shape(graph['paper'].label) == [4019, 3]
    assert len(graph['feat_p/a/s']) == 3
    assert len(graph['metapath']) == 2
    assert len(graph['train']) == 3
    assert len(graph['val']) == 3
    assert len(graph['test']) == 3


# test_acm()
