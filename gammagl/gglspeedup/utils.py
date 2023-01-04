import numpy as np



def reindex_by_config(adj_csr, graph_feature):

    node_count = adj_csr.indptr.shape[0] - 1

    # sort and shuffle
    degree = adj_csr.indptr[1:] - adj_csr.indptr[:-1]
    prev_order = np.argsort(-degree)
    range_ = np.arange(node_count, dtype=np.int64)
    new_order = np.empty_like(range_)
    new_order[prev_order] = range_
    graph_feature = graph_feature[prev_order]
    return graph_feature, new_order


def reindex_feature(adj_csr, feature):
    feature, new_order = reindex_by_config(adj_csr, feature)
    return feature, new_order

UNITS = {
    #
    "KB": 2**10,
    "MB": 2**20,
    "GB": 2**30,
    #
    "K": 2**10,
    "M": 2**20,
    "G": 2**30,
}

def parse_size(sz) -> int:
    if isinstance(sz, int):
        return sz
    elif isinstance(sz, float):
        return int(sz)
    elif isinstance(sz, str):
        for suf, u in sorted(UNITS.items()):
            if sz.upper().endswith(suf):
                return int(float(sz[:-len(suf)]) * u)
    raise Exception("invalid size: {}".format(sz))



