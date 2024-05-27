import tensorlayerx as tlx
import numpy as np


def get_train_val_test_split(dataset, train_per_class, val_per_class):
    """Split the dataset into train, validation, and test sets.

    Parameters
    ----------
    dataset :
        The dataset to split.
    train_per_class : int
        The number of training examples per class.
    val_per_class : int
        The number of validation examples per class.

    Returns
    -------
    :class:`tuple` of :class:`tensor`
    
    """
    graph = dataset[0]
    random_state = np.random.RandomState(0)
    labels = tlx.nn.OneHot(depth=dataset.num_classes)(graph.y).numpy()
    num_samples, num_classes = graph.num_nodes, dataset.num_classes
    remaining_indices = set(range(num_samples))
    forbidden_indices = set()

    train_indices = sample_per_class(random_state, num_samples, num_classes, labels, train_per_class, forbidden_indices=forbidden_indices)
    forbidden_indices.update(train_indices)
    val_indices = sample_per_class(random_state, num_samples, num_classes, labels, val_per_class, forbidden_indices=forbidden_indices)
    forbidden_indices.update(val_indices)
    test_indices = np.array(list(remaining_indices - forbidden_indices))

    return generate_masks(graph.num_nodes, train_indices, val_indices, test_indices)


def sample_per_class(random_state, num_samples, num_classes, labels, num_examples_per_class, forbidden_indices=None):
    sample_indices_per_class = {index: [] for index in range(num_classes)}
    forbidden_set = set(forbidden_indices) if forbidden_indices is not None else set()

    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0 and sample_index not in forbidden_set:
                sample_indices_per_class[class_index].append(sample_index)

    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(num_classes)
         ])


def generate_masks(num_nodes, train_indices, val_indices, test_indices): 
    np_train_mask = np.zeros(num_nodes) 
    np_train_mask[train_indices] = 1 
    np_val_mask = np.zeros(num_nodes) 
    np_val_mask[val_indices] = 1 
    np_test_mask = np.zeros(num_nodes) 
    np_test_mask[test_indices] = 1 
    train_mask = tlx.ops.convert_to_tensor(np_train_mask, dtype=tlx.bool) 
    val_mask = tlx.ops.convert_to_tensor(np_val_mask, dtype=tlx.bool)
    test_mask = tlx.ops.convert_to_tensor(np_test_mask, dtype=tlx.bool) 
    return train_mask, val_mask, test_mask