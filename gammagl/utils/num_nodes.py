from copy import copy
from typing import Optional
import tensorlayerx as tlx
def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif tlx.is_tensor(edge_index):
        return int(tlx.reduce_max(edge_index)) + 1 if edge_index is not None else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))
