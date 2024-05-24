import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR


import tensorlayerx as tlx
import numpy as np



def sample_per_class(random_state, num_samples, num_classes, labels, num_examples_per_class, forbidden_indices=None):
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def get_train_val_test_split(dataset, train_per_class, val_per_class):
    graph = dataset[0]
    random_state = np.random.RandomState(0)
    train_examples_per_class = train_per_class,
    val_examples_per_class = val_per_class,
    labels = tlx.nn.OneHot(depth=dataset.num_classes)(graph.y).numpy()
    num_samples, num_classes = graph.num_nodes, dataset.num_classes
    remaining_indices = list(range(num_samples))
    forbidden_indices = []
    train_indices = sample_per_class(random_state, num_samples, num_classes, labels, train_examples_per_class, forbidden_indices=forbidden_indices)
    forbidden_indices = np.concatenate((forbidden_indices, train_indices))
    val_indices = sample_per_class(random_state, num_samples, num_classes, labels, val_examples_per_class, forbidden_indices=forbidden_indices)
    forbidden_indices = np.concatenate((forbidden_indices, val_indices))
    test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    np_train_mask = np.zeros(graph.num_nodes)
    np_train_mask[train_indices] = 1
    np_val_mask = np.zeros(graph.num_nodes)
    np_val_mask[val_indices] = 1
    np_test_mask = np.zeros(graph.num_nodes)
    np_test_mask[test_indices] = 1

    train_mask = tlx.ops.convert_to_tensor(np_train_mask, dtype=tlx.bool)
    val_mask = tlx.ops.convert_to_tensor(np_val_mask, dtype=tlx.bool)
    test_mask = tlx.ops.convert_to_tensor(np_test_mask, dtype=tlx.bool)

    return train_mask, val_mask, test_mask