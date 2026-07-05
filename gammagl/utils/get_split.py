import tensorlayerx as tlx
import numpy as np
from sklearn.model_selection import train_test_split


def get_train_val_test_split(graph, train_ratio, val_ratio):
    """
    Split the dataset into train, validation, and test sets.

    Parameters
    ----------
    graph :
        The graph to split.
    train_ratio : float
        The proportion of the dataset to include in the train split.
    val_ratio : float
        The proportion of the dataset to include in the validation split.

    Returns
    -------
    :class:`tuple` of :class:`tensor`
    """

    random_state = np.random.RandomState(0)
    num_samples = graph.num_nodes
    all_indices = np.arange(num_samples)

    # split into train and (val + test)
    train_indices, val_test_indices = train_test_split(
        all_indices, train_size=train_ratio, random_state=random_state
    )

    # calculate the ratio of validation and test splits in the remaining data
    test_ratio = 1.0 - train_ratio - val_ratio
    val_size_ratio = val_ratio / (val_ratio + test_ratio)

    # split val + test into validation and test sets
    val_indices, test_indices = train_test_split(
        val_test_indices, train_size=val_size_ratio, random_state=random_state
    )

    return generate_masks(num_samples, train_indices, val_indices, test_indices)


def generate_masks(num_nodes, train_indices, val_indices, test_indices):
    np_train_mask = np.zeros(num_nodes, dtype=bool)
    np_train_mask[train_indices] = 1
    np_val_mask = np.zeros(num_nodes, dtype=bool)
    np_val_mask[val_indices] = 1
    np_test_mask = np.zeros(num_nodes, dtype=bool)
    np_test_mask[test_indices] = 1

    train_mask = tlx.ops.convert_to_tensor(np_train_mask, dtype=tlx.bool)
    val_mask = tlx.ops.convert_to_tensor(np_val_mask, dtype=tlx.bool)
    test_mask = tlx.ops.convert_to_tensor(np_test_mask, dtype=tlx.bool)

    return train_mask, val_mask, test_mask


def get_few_shot_split(labels, num_shots, test_ratio=0.2, random_state=0):
    """Sample a minimal few-shot train/test split for node classification.

    This follows the original EdgePrompt node downstream protocol closely:
    sample up to ``num_shots`` nodes per class for training, remove those nodes
    from the candidate pool, then draw a random test subset from the remaining
    nodes.
    """
    if test_ratio <= 0 or test_ratio > 1:
        raise ValueError('test_ratio must be in (0, 1].')

    labels = tlx.reshape(labels, (-1,))
    labels_np = tlx.convert_to_numpy(labels)
    rng = np.random.RandomState(random_state)

    train_indices = []
    for cls in np.unique(labels_np):
        cls_indices = np.where(labels_np == cls)[0]
        if cls_indices.shape[0] <= num_shots:
            train_indices.extend(cls_indices.tolist())
        else:
            train_indices.extend(rng.choice(cls_indices, size=num_shots, replace=False).tolist())

    train_set = set(train_indices)
    remaining_indices = [idx for idx in rng.permutation(labels_np.shape[0]).tolist() if idx not in train_set]
    num_test = max(1, int(test_ratio * labels_np.shape[0]))
    num_test = min(num_test, len(remaining_indices))
    test_indices = remaining_indices[:num_test]

    return (
        tlx.convert_to_tensor(np.asarray(train_indices, dtype=np.int64), dtype=tlx.int64),
        tlx.convert_to_tensor(np.asarray(test_indices, dtype=np.int64), dtype=tlx.int64),
    )
