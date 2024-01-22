import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR
import argparse
import numpy as np
import tensorlayerx as tlx
import logging

tlx_logger = logging.getLogger("tensorlayerx")
tlx_logger.setLevel(logging.ERROR)
from distill import model_train
from gammagl.datasets import Planetoid
import yaml
from teacher_trainer import teacher_trainer
from gammagl.models import GCNModel, GATModel
from gammagl.utils import remove_self_loops
import scipy.sparse as sp
import torch


def get_training_config(config_path, args):
    with open(config_path, 'r') as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    configs = dict(
        args.__dict__, **full_config[args.model][args.dataset])
    return configs


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


def get_train_val_test_split(graph, configs):
    random_state = np.random.RandomState(0)
    train_examples_per_class = configs['train_per_class'],
    val_examples_per_class = configs['val_per_class'],
    my_val_per_class = configs['my_val_per_class']
    labels = tlx.nn.OneHot(depth=configs['num_classes'])(graph.y).numpy()
    num_samples, num_classes = graph.num_nodes, configs['num_classes']
    remaining_indices = list(range(num_samples))
    forbidden_indices = []
    train_indices = sample_per_class(random_state, num_samples, num_classes, labels, train_examples_per_class, forbidden_indices=forbidden_indices)
    forbidden_indices = np.concatenate((forbidden_indices, train_indices))
    val_indices = sample_per_class(random_state, num_samples, num_classes, labels, val_examples_per_class, forbidden_indices=forbidden_indices)
    forbidden_indices = np.concatenate((forbidden_indices, val_indices))
    my_val_indices = sample_per_class(random_state, num_samples, num_classes, labels, my_val_per_class, forbidden_indices=forbidden_indices)
    forbidden_indices = np.concatenate((forbidden_indices, my_val_indices))
    test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    np_train_mask = np.zeros(graph.num_nodes)
    np_train_mask[train_indices] = 1
    np_val_mask = np.zeros(graph.num_nodes)
    np_val_mask[val_indices] = 1
    np_test_mask = np.zeros(graph.num_nodes)
    np_test_mask[test_indices] = 1
    np_my_val_mask = np.zeros(graph.num_nodes)
    np_my_val_mask[my_val_indices] = 1

    train_mask = tlx.ops.convert_to_tensor(np_train_mask).bool()
    val_mask = tlx.ops.convert_to_tensor(np_val_mask).bool()
    my_val_mask = tlx.ops.convert_to_tensor(np_my_val_mask).bool()
    test_mask = tlx.ops.convert_to_tensor(np_test_mask).bool()

    return train_mask, val_mask, my_val_mask, test_mask


def edge_index_to_csr_matrix(edge_index, num_nodes):
    coo_matrix = sp.coo_matrix((torch.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                               shape=(num_nodes, num_nodes), dtype=float)
    csr_matrix = coo_matrix.tocsr()

    return csr_matrix


def csr_matrix_to_edge_index(csr_matrix):
    coo_matrix = csr_matrix.tocoo()
    row_indices = torch.tensor(coo_matrix.row, dtype=torch.long)
    col_indices = torch.tensor(coo_matrix.col, dtype=torch.long)
    edge_index = torch.stack([row_indices, col_indices], dim=0)

    return edge_index


def create_subgraph(graph, nodes_to_keep=None):
    if nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    new_adj_matrix = edge_index_to_csr_matrix(graph.edge_index, graph.num_nodes)[nodes_to_keep][:, nodes_to_keep]
    graph.edge_index = csr_matrix_to_edge_index(new_adj_matrix)
    graph.x = graph.x[nodes_to_keep]
    graph.y = graph.y[nodes_to_keep]
    graph.num_nodes = tlx.get_tensor_shape(graph.x)[0]
    return graph


def get_largest_connected_component(graph):
    edge_index_no_self_loops, _ = remove_self_loops(graph.edge_index)
    graph.edge_index = edge_index_no_self_loops
    _, component_indices = sp.csgraph.connected_components(
        edge_index_to_csr_matrix(graph.edge_index, graph.num_nodes))
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:1]
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    graph = create_subgraph(graph, nodes_to_keep)
    return graph


def load_dataset(configs):
    if str.lower(configs['dataset']) in ['cora', 'pubmed', 'citeseer']:
        dataset = Planetoid(configs['dataset_path'], configs['dataset'])
        dataset.process()
        graph = dataset[0]
    else:
        raise ValueError('Unknown dataset: {}'.format(configs['dataset']))
    if configs['largest_connected_component']:
        graph = get_largest_connected_component(graph)
    graph_configs = {
        'num_node_features': dataset.num_node_features,
        'num_classes': dataset.num_classes,
    }
    configs = dict(configs, **graph_configs)
    train_mask, val_mask, my_val_mask, graph.test_mask = get_train_val_test_split(graph, configs)

    dataset_configs = {
        'train_mask': train_mask,
        'val_mask': val_mask,
        'my_val_mask': my_val_mask,
        't_train_mask': graph.train_mask,
        't_val_mask': graph.val_mask,
        't_test_mask': graph.test_mask
    }
    configs = dict(
        configs, **dataset_configs)
    return configs, graph


def choose_model(configs):
    if configs['model'] == 'GCN':
        model = GCNModel(feature_dim=configs['num_node_features'],
                         hidden_dim=configs['hidden_dim'],
                         num_class=configs['num_classes'],
                         drop_rate=configs['drop_rate'],
                         num_layers=configs['num_layers'],
                         norm='both',
                         name='GCN')
    elif configs['model'] == 'GAT':
        model = GATModel(feature_dim=configs['num_node_features'],
                         hidden_dim=configs['hidden_dim'],
                         num_class=configs['num_classes'],
                         heads=configs['heads'],
                         drop_rate=configs['drop_rate'],
                         num_layers=configs['num_layers'],
                         name="GAT",
                         )
    else:
        raise ValueError(f'Undefined Model.')
    return model


def main(args):
    configs = get_training_config("train_conf.yaml", args)
    configs, graph = load_dataset(configs)
    teacher_model = choose_model(configs)
    teacher_logits, acc_teacher_test = teacher_trainer(graph, configs, teacher_model)
    student_model = choose_model(configs)
    best_acc_test = model_train(configs, student_model, graph, teacher_logits)
    print(
        "acc_student_test: {:.4f}  acc_teacher_test: {:.4f}".format(best_acc_test.item(), acc_teacher_test.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    parser.add_argument('--model', type=str,
                        default='GAT', help='Teacher and student Model')
    parser.add_argument("--max_epoch", type=int, default=1600, help="max number of epoch")
    parser.add_argument("--patience", type=int, default=800, help="early stopping epoch")
    parser.add_argument('--largest_connected_component', type=bool, default=True,
                        help='use largest connected component or not')
    args = parser.parse_args()

    main(args)
