import os

os.environ['TL_BACKEND'] = 'torch'
import argparse
from pathlib import Path
import numpy as np
import tensorlayerx as tlx
from distill import choose_model, model_train, layer_normalize
from gammagl.datasets import Planetoid

predefined_configs = {
    'GCN': {
        'cora': {
            'train_per_class': 20,
            'val_per_class': 20,
            'my_val_per_class': 10,
            'num_test': 2358,
            'hidden_dim': 64,
            'drop_rate': 0.8,
            'num_layers': 2,
            'my_lr': 0.0010048137731444346,
            'my_t_lr': 0.00013337740190874308,
            'lam': 1,  # 正则化参数
            'k': 1
        },
        'citeseer': {
            'train_per_class': 20,
            'val_per_class': 20,
            'my_val_per_class': 10,
            'num_test': 3000,
            'hidden_dim': 64,
            'drop_rate': 0.8,
            'num_layers': 2,
            'my_lr': 0.0008736877838717051,
            'my_t_lr': 0.0019445362780140703,
            'lam': 1,
            'k': 1
        },
        'pubmed': {
            'train_per_class': 20,
            'val_per_class': 20,
            'my_val_per_class': 10,
            'num_test': 19000,
            'hidden_dim': 64,
            'drop_rate': 0.8,
            'num_layers': 2,
            'my_lr': 7.39520400332059E-07,
            'my_t_lr': 0.0000937937968682644,
            'lam': 1000,
            'k': 1
        },
    },
    'GAT': {
        'cora': {
            'train_per_class': 20,
            'val_per_class': 20,
            'my_val_per_class': 10,
            'num_test': 2358,
            'hidden_dim': 8,
            'heads': 8,
            'drop_rate': 0.6,
            'num_layers': 2,
            'my_lr': 0.0012033602084655297,
            'my_t_lr': 0.0009098411162144382,
            'lam': 1,
            'k': 2
        },
        'citeseer': {
            'train_per_class': 20,
            'val_per_class': 20,
            'my_val_per_class': 10,
            'num_test': 3000,
            'hidden_dim': 8,
            'heads': 8,
            'drop_rate': 0.6,
            'num_layers': 2,
            'my_lr': 0.000858867483786378,
            'my_t_lr': 0.00156837256987387,
            'lam': 1,
            'k': 2
        },
        'pubmed': {
            'train_per_class': 20,
            'val_per_class': 20,
            'my_val_per_class': 10,
            'num_test': 19000,
            'hidden_dim': 8,
            'heads': 8,
            'drop_rate': 0.6,
            'num_layers': 2,
            'my_lr': 0.0000295072855477947,
            'my_t_lr': 0.000003722992284806,
            'lam': 200,
            'k': 2
        },
    },
}


def choose_path(conf):
    cascade_dir = Path.cwd().joinpath('outputs', conf['dataset'], conf['teacher'],
                                      'logits.npy')
    return cascade_dir


def load_teacher_logits(cascade_dir):
    loaded_logits_array = np.load(cascade_dir)
    loaded_logits = tlx.ops.convert_to_tensor(loaded_logits_array)
    return loaded_logits


def split_train_mask(graph, num_classes, train_per_class, val_per_class, my_val_per_class):
    label = graph.y[graph.train_mask].numpy()
    indices = np.where(graph.train_mask.numpy())[0]
    class_indices = [indices[np.where(label == c)[0]] for c in range(num_classes)]
    np_train_mask = np.zeros(graph.num_nodes)
    np_val_mask = np.zeros(graph.num_nodes)
    np_my_val_mask = np.zeros(graph.num_nodes)
    for i in range(num_classes):
        np.random.shuffle(class_indices[i])
        np_train_mask[class_indices[i][:train_per_class]] = 1
        np_val_mask[class_indices[i][train_per_class:train_per_class + val_per_class]] = 1
        np_my_val_mask[
            class_indices[i][train_per_class + val_per_class:train_per_class + val_per_class + my_val_per_class]] = 1
    train_mask = tlx.ops.convert_to_tensor(np_train_mask).bool()
    val_mask = tlx.ops.convert_to_tensor(np_val_mask).bool()
    my_val_mask = tlx.ops.convert_to_tensor(np_my_val_mask).bool()
    return train_mask, val_mask, my_val_mask


def load_dataset(configs):
    if str.lower(configs['dataset']) in ['cora', 'pubmed', 'citeseer']:
        dataset = Planetoid(configs['dataset_path'], configs['dataset'], split="random",
                            num_train_per_class=configs['train_per_class'] + configs['val_per_class'] + configs[
                                'my_val_per_class'],
                            num_val=0, num_test=configs['num_test'])
        dataset.process()
        graph = dataset[0]
    else:
        raise ValueError('Unknown dataset: {}'.format(configs['dataset']))
    train_mask, val_mask, my_val_mask = split_train_mask(graph, dataset.num_classes, configs['train_per_class'],
                                                         configs['val_per_class'], configs['my_val_per_class'])
    dataset_configs = {
        'num_node_features': dataset.num_node_features,
        'num_classes': dataset.num_classes,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'my_val_mask': my_val_mask
    }
    configs = dict(
        configs, **dataset_configs)
    return configs, graph


def main(args):
    configs = dict(
        args.__dict__, **predefined_configs[args.teacher][args.dataset])
    configs, graph = load_dataset(configs)
    cascade_dir = choose_path(configs)
    teacher_logits = load_teacher_logits(cascade_dir)
    teacher_logits = layer_normalize(teacher_logits)
    model = choose_model(configs)
    best_acc_test = model_train(configs, model, graph, teacher_logits)
    print(
        "best acc test: {:.4f} ".format(best_acc_test.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument('--teacher', type=str,
                        default='GCN', help='Teacher Model')
    parser.add_argument('--student', type=str,
                        default='GCN', help='Student Model')
    parser.add_argument("--max_epoch", type=int, default=300, help="max number of epoch")
    parser.add_argument("--patience", type=int, default=50, help="early stopping epoch")
    args = parser.parse_args()

    main(args)
