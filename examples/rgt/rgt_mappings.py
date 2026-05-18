import numpy as np
import json

class_maps = {
    'cora': {
        1: list(range(2)),
        2: list(range(2, 4)),
        3: list(range(4, 7))
    },
    'citeseer': {
        1: list(range(2)),
        2: list(range(2, 4)),
        3: list(range(4, 6))
    },
    'pubmed': {
        1: list(range(2)),
        2: list(range(2, 4)),
        3: []
    },
    'computers': {
        1: list(range(10)),
        2: list(range(0, 5)),
        3: list(range(0, 10))
    },
    'physics': {
        1: list(range(5)),
        2: list(range(0, 10)),
        3: list(range(0, 10))
    },
    'photo': {
        1: list(range(5)),
        2: list(range(0, 5)),
        3: list(range(0, 10))
    },
    'arxiv': {
        1: list(range(20)),
        2: list(range(0, 5)),
        3: list(range(0, 10)),
        4: list(range(0, 15)),
        5: list(range(0, 15)),
        6: list(range(0, 10))
    },
    'github': {
        1: list(range(2, 4)),
        2: list(range(0, 10)),
        3: list(range(5, 15))
    },
    'deezer': {
        1: list(range(15)),
        2: list(range(0, 5)),
        3: list(range(0, 10)),
        4: list(range(0, 15)),
        5: list(range(0, 15)),
        6: list(range(0, 10))
    }
}

def map_to_label(mapping, predictions):
    pass
