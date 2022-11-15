import numpy as np


def read_embeddings(filename, n_node, n_embed):
    """read pre trained and learned node embeddings
    """

    with open(filename, "r") as f:
        lines = f.readlines()
        embedding_matrix = np.random.rand(n_node, n_embed)
        for line in lines:
            emd = line.split()
            embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
        return embedding_matrix


def str_list_to_float(str_list):
    return [float(item) for item in str_list]
