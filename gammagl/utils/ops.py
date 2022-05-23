

def get_index_from_counts(counts):
    """Return index generated from counts
    This function return the index from given counts.
    For example, when counts = [ 2, 3, 4], return [0, 2, 5, 9]

    Parameters
    ----------
    counts: numpy.ndarray of paddle.Tensor

    Returns
    -------
        Return idnex of the counts
    """
    if check_is_tensor(counts):
        index = paddle.concat(
            [
                paddle.zeros(
                    shape=[1, ], dtype=counts.dtype), paddle.cumsum(counts)
            ],
            axis=-1)
    else:
        index = np.cumsum(counts, dtype="int64")
        index = np.insert(index, 0, 0)
    return index
