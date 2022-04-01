from copy import copy
from typing import Optional
import tensorlayerx as tlx
def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif tlx.is_tensor(edge_index):
        return max(int(tlx.reduce_max(edge_index[0],axis=0)),
                   int(tlx.reduce_max(edge_index[1],axis=0)))\
               + 1 if edge_index is not None else 0
    else:
        return max(max(edge_index[0]),max(edge_index[0]))
