import os

os.environ['TL_BACKEND'] = 'torch'
import argparse
import numpy as np
import tensorlayerx as tlx
import logging

tlx_logger = logging.getLogger("tensorlayerx")
tlx_logger.setLevel(logging.ERROR)
from distill import choose_model, model_train, layer_normalize
from gammagl.datasets import Planetoid
import yaml
from teacher_trainer import teacher_trainer


def get_training_config(config_path, args):
    with open(config_path, 'r') as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    configs = dict(
        args.__dict__, **full_config[args.teacher][args.dataset])
    return configs


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
    configs = get_training_config("train_conf.yaml", args)
    configs, graph = load_dataset(configs)
    teacher_logits, acc_teacher_test = teacher_trainer(configs)
    teacher_logits = layer_normalize(teacher_logits)
    model = choose_model(configs)
    best_acc_test = model_train(configs, model, graph, teacher_logits)
    print(
        "acc_student_test: {:.4f}  acc_teacher_test: {:.4f}".format(best_acc_test.item(), acc_teacher_test.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument('--teacher', type=str,
                        default='GAT', help='Teacher Model')
    parser.add_argument('--student', type=str,
                        default='GAT', help='Student Model')
    parser.add_argument("--max_epoch", type=int, default=300, help="max number of epoch")
    parser.add_argument("--patience", type=int, default=200, help="early stopping epoch")
    args = parser.parse_args()

    main(args)
